"""
Fetch a guide document from the Drive folder
and append its full text to drive_structure.txt as context.
"""

import io
import os
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
OUTPUT_FILE = PROJECT_ROOT / "data" / "drive_structure.txt"
GUIDE_NAME_FRAGMENT = "Program Guide"  # Set to a filename fragment of your guide document


def find_key_file() -> str:
    secrets_dir = PROJECT_ROOT / "secrets"
    matches = list(secrets_dir.glob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON key file found in {secrets_dir}")
    return str(matches[0])


def build_drive_service():
    key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY") or find_key_file()
    creds = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def find_guide(service, drive_id: str) -> dict:
    resp = service.files().list(
        corpora="drive",
        driveId=drive_id,
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
        q=f"name contains '{GUIDE_NAME_FRAGMENT}' and trashed = false",
        fields="files(id, name, mimeType)",
    ).execute()
    files = resp.get("files", [])
    if not files:
        raise FileNotFoundError(f"No file found matching '{GUIDE_NAME_FRAGMENT}'")
    if len(files) > 1:
        print(f"Multiple matches — using first: {files[0]['name']}")
    return files[0]


def extract_text(service, file: dict) -> str:
    mime = file["mimeType"]
    file_id = file["id"]

    # Google Docs — export as plain text
    if mime == "application/vnd.google-apps.document":
        resp = service.files().export(
            fileId=file_id, mimeType="text/plain"
        ).execute()
        return resp.decode("utf-8") if isinstance(resp, bytes) else resp

    # PDF
    if mime == "application/pdf":
        import pdfplumber
        buf = io.BytesIO()
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        pages = []
        with pdfplumber.open(buf) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
        return "\n\n".join(pages)

    # DOCX
    if mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        import docx
        buf = io.BytesIO()
        request = service.files().get_media(fileId=file_id)
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        buf.seek(0)
        doc = docx.Document(buf)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    raise ValueError(f"Unsupported MIME type: {mime}")


def main():
    drive_id = os.getenv("SHARED_DRIVE_ID")
    if not drive_id:
        raise EnvironmentError("SHARED_DRIVE_ID is not set in .env")

    print("Connecting to Google Drive...")
    service = build_drive_service()

    print(f"Searching for '{GUIDE_NAME_FRAGMENT}'...")
    guide = find_guide(service, drive_id)
    print(f"Found: {guide['name']} ({guide['mimeType']})")

    print("Extracting text...")
    text = extract_text(service, guide)

    divider = "=" * 60
    section = (
        f"\n\n{divider}\n"
        f"PROGRAM GUIDE\n"
        f"Source: {guide['name']}\n"
        f"{divider}\n\n"
        f"{text.strip()}\n"
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(section)

    print(f"Appended to {OUTPUT_FILE} ({len(text):,} characters)")


if __name__ == "__main__":
    main()
