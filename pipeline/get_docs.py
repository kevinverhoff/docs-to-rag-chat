"""
Step 2: Document Fetcher

Downloads every file from the Shared Drive and writes:
  - data/raw/{file_id}.{ext}   one file per document
  - metadata.json              one record per file with structured metadata

Metadata fields parsed from folder paths:
  program         Top-level program area (SWS, Focus K-3, etc.)
  doc_type        Document category (site_visit, survey_reflection, etc.)
  academic_year   e.g. "2025-26" -- derived using the July boundary rule:
                    month >= 7  ->  [year]-[year+1]
                    month < 7   ->  [year-1]-[year]
  season          "Fall" (Jul-Dec) or "Spring" (Jan-Jun), null if unknown
  date_precision  "direct" | "month_derived" | "unknown"
  district        Lake, Lee, Osceola, Pasco, Polk, St. Lucie, etc.

Usage:
  python pipeline/get_docs.py

Requires .env:
  SHARED_DRIVE_ID=...

And secrets/*.json (service account key).
"""

import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")
DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_PATH = PROJECT_ROOT / "data" / "metadata.json"
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
DOWNLOAD_DELAY_SECONDS = 0.1

# ---------------------------------------------------------------------------
# Program mapping -- keyed on lowercase folder name
# ---------------------------------------------------------------------------

PROGRAM_MAP: dict[str, str] = {
    "!_multiple programs": "Multiple Programs",
    "0_sws": "SWS",
    "1_focus k-3": "Focus K-3",
    "2_math materials": "Math Materials",
    "3_teacher workforce": "Teacher Workforce",
    "4_eir/game based learning": "EIR/GBL",
    "4_eir": "EIR/GBL",
    "5_spark": "SPARK",
    "6_nat hqim/policy & advocacy": "NAT HQIM",
    "6_nat hqim": "NAT HQIM",
    "background": "Background",
}

# ---------------------------------------------------------------------------
# District names -- lowercase for substring matching against path + filename
# Ordered longest-first so "palm beach" matches before "palm"
# ---------------------------------------------------------------------------

KNOWN_DISTRICTS: list[str] = [
    "miami-dade", "st. lucie", "palm beach", "santa rosa",
    "hillsborough", "okaloosa", "alachua", "broward", "osceola",
    "flagler", "collier", "marion", "nassau", "putnam",
    "pasco", "lake", "levy", "clay", "polk", "lee",
]

# Canonical display names keyed on the lowercase match string above
DISTRICT_DISPLAY: dict[str, str] = {d: d.title() for d in KNOWN_DISTRICTS}
DISTRICT_DISPLAY["miami-dade"] = "Miami-Dade"
DISTRICT_DISPLAY["st. lucie"] = "St. Lucie"

# ---------------------------------------------------------------------------
# Month name to number
# ---------------------------------------------------------------------------

MONTH_MAP: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

# ---------------------------------------------------------------------------
# Doc-type keyword rules -- first match wins
# ---------------------------------------------------------------------------

# Folder-path-based rules (matched against folder_path)
# ---------------------------------------------------------------------------
# Derived metadata lookups
# ---------------------------------------------------------------------------

FILE_TYPE_MAP: dict[str, str] = {
    "application/pdf": "pdf",
    "application/vnd.google-apps.document": "document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "document",
    "application/msword": "document",
    "application/vnd.google-apps.presentation": "presentation",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "presentation",
    "application/vnd.ms-powerpoint": "presentation",
    "application/vnd.google-apps.spreadsheet": "spreadsheet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "spreadsheet",
    "application/vnd.ms-excel": "spreadsheet",
    "text/csv": "spreadsheet",
}

# data_form is a schema field populated by extract_themes.py via LLM inference
# over document content — not derived here from doc_type.

DOC_TYPE_RULES: list[tuple[str, list[str]]] = [
    ("program_content",           ["!_programming"]),
    ("intake_survey",             ["intake", "application", "pre-survey"]),
    ("feedback_survey_data",      ["survey reflection", "eoy survey", "end of year",
                                   "convening data", "convening feedback",
                                   "network survey", "event survey", "perts"]),
    ("qualitative_theming",       ["listening session", "focus group"]),
    ("engagement_notes",          ["site visit"]),
    ("evaluation_report",         ["evaluation report", "westedreport", "wested"]),
    ("district_artifact",         ["district artifact", "district data",
                                   "district strengths", "fldoe"]),
    ("teacher_practice_data",     ["podcast", "what counts"]),
    ("field_influence",           ["policy", "hqim planning"]),
    ("grants_and_funder_reporting", ["annual performance", "performance report",
                                   "virtual call", "gcit", "mid-year reflection"]),
    ("program_overview",          ["planning document", "spark planning",
                                   "concept and district"]),
    ("progress_summary",          ["progress summar"]),
    ("other_data_file",           ["baseline data", "tw baseline"]),
    ("program_logistics",         ["readiness data"]),
]

# Filename-based rules — take precedence over folder rules (matched against file_name)
# More specific signal: filename patterns that reliably identify a type regardless of folder
DOC_TYPE_FILENAME_RULES: list[tuple[str, list[str]]] = [
    # program_logistics — admin/tracking files that land in intake folders
    ("program_logistics",              ["tracker", "master participant database",
                                        "alumni information", "application acceptance",
                                        "emails of", "teacher cadre.xlsx",
                                        "coaching log"]),
    # program_engagement_notes — notes files that land in site_visit/convening folders
    ("engagement_notes",       ["meeting notes", "huddle tool",
                                        "breakout room notes"]),
    # program_progress_summary — synthesis/summary docs in site_visit or district folders
    ("progress_summary",       ["learning walk summary", "learning walk synthesis",
                                        "cadre summary", "cadre summaries",
                                        "implementation pattern summary",
                                        "perceptions and use of", "perspectives on instructional"]),
    # feedback_survey_data — raw survey response files across convening, site_visit, district folders
    ("feedback_survey_data",           ["feedback survey responses", "exit ticket",
                                        "slido"]),
    # qualitative_summaries — synthesized qualitative analysis docs (PDFs/docs only, not raw xlsx)
    ("qualitative_theming",          ["focus groups report", "empathy interview report",
                                        "listening sessions report", "listening session themes",
                                        "focus group summary", "stakeholder feedback synthesis",
                                        "feedback synthesis.pdf"]),
    # teacher_practice_survey_data — cycle/survey reflection forms and site visit surveys
    ("teacher_practice_data",   ["cycle a reflection", "cycle b reflection",
                                        "cycle c reflection", "cycle d reflection",
                                        "cycle e reflection", "reflection form (responses)",
                                        "reflection form responses",
                                        "survey reflections.docx",
                                        "data and reflections.docx",
                                        "site visit survey", "site visits survey",
                                        "kickoff survey"]),
    # impact_report — polished IF-authored results docs
    ("impact_data_report",                  ["impact brief", "impact report", "results slides",
                                        "year 2 report", "year 3 report",
                                        "year 4 report", "year 5 report"]),
    # impact_data_reports — external evaluator data visualizations and slide decks
    ("impact_data_report",            ["visualizations from wested",
                                        "perts graphs from wested",
                                        "teacher-level visualizations",
                                        "data slide deck", "data report"]),
    # other_data_files — raw quantitative data files
    ("other_data_file",               ["grade-level visualizations from wested",
                                        "underlying data"]),
    # evaluation_report — third-party research studies
    ("evaluation_report",              ["usability study", "feasibility study report",
                                        "registry of efficacy"]),
    # program_overview — orientation/description docs
    ("program_overview",               ["district overview doc", "program concept",
                                        "one pager"]),
    # grant — funder updates, gates trackers, check-in notes
    ("grants_and_funder_reporting",                          ["funder check-in", "funder check in",
                                        "update to gates", "gates results tracker"]),
    # program_logistics — results trackers
    ("program_logistics",              ["results tracker"]),
    # program_progress_summary — monthly progress summaries
    ("progress_summary",       ["monthly progress summar"]),
]

# ---------------------------------------------------------------------------
# Google-native export targets
# ---------------------------------------------------------------------------

GOOGLE_EXPORT_MIME: dict[str, tuple[str, str]] = {
    "application/vnd.google-apps.document": (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".docx",
    ),
    "application/vnd.google-apps.spreadsheet": (
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".xlsx",
    ),
    "application/vnd.google-apps.presentation": (
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
    ),
    "application/vnd.google-apps.form": (
        "application/pdf",
        ".pdf",
    ),
}

MIME_TO_EXT: dict[str, str] = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.ms-powerpoint": ".ppt",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}

# MIME types that cannot be meaningfully downloaded or exported
SKIP_MIME: set[str] = {
    "application/vnd.google-apps.shortcut",
    "application/vnd.google-apps.map",
    "application/vnd.google-apps.script",
}

# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

def _find_key_file() -> str:
    secrets_dir = PROJECT_ROOT / "secrets"
    matches = list(secrets_dir.glob("*.json"))
    if not matches:
        raise FileNotFoundError(f"No JSON key file found in {secrets_dir}")
    if len(matches) > 1:
        print(f"Warning: multiple key files in secrets/, using {matches[0].name}")
    return str(matches[0])


def build_drive_service():
    key_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_KEY") or _find_key_file()
    creds = service_account.Credentials.from_service_account_file(key_path, scopes=SCOPES)
    return build("drive", "v3", credentials=creds)


def list_all_items(service, drive_id: str) -> list[dict]:
    """Return every file and folder in the Shared Drive."""
    items = []
    page_token = None
    while True:
        resp = service.files().list(
            corpora="drive",
            driveId=drive_id,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
            q="trashed = false",
            fields="nextPageToken, files(id, name, mimeType, parents, webViewLink)",
            pageSize=1000,
            pageToken=page_token,
        ).execute()
        items.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return items


def build_path_map(items: list[dict], drive_id: str) -> dict[str, str]:
    """
    Map every item ID to its full slash-separated folder path.
    Uses iterative BFS so deeply-nested structures do not hit Python recursion limits.
    """
    id_to_item = {item["id"]: item for item in items}
    cache: dict[str, str] = {drive_id: ""}

    pending = list(id_to_item.keys())

    for _ in range(30):  # max folder depth guard
        if not pending:
            break
        still_pending = []
        for item_id in pending:
            item = id_to_item[item_id]
            parents = item.get("parents", [])
            if not parents:
                cache[item_id] = item.get("name", "")
                continue
            parent_id = parents[0]
            if parent_id in cache:
                parent_path = cache[parent_id]
                name = item.get("name", "")
                cache[item_id] = f"{parent_path}/{name}" if parent_path else name
            else:
                still_pending.append(item_id)
        pending = still_pending

    # Orphaned items (no resolved parent) fall back to bare name
    for item_id in pending:
        cache[item_id] = id_to_item[item_id].get("name", item_id)

    return cache

# ---------------------------------------------------------------------------
# Metadata parsing
# ---------------------------------------------------------------------------

def _parse_academic_year(folder_name: str) -> tuple[str | None, str | None, str]:
    """
    Extract (academic_year, season, date_precision) from a folder name.

    Academic year boundary: July (month 7).
      month >= 7  ->  academic_year = "{year}-{(year+1) % 100:02d}"
      month < 7   ->  academic_year = "{year-1}-{year % 100:02d}"
    """
    name = folder_name.strip()

    # Direct academic year: "2022-23", "2024-25", "2023-2024"
    # Validate that end is exactly one year after start to avoid matching ISO dates like "2026-05-26"
    m = re.search(r'\b(20\d{2})[-](\d{2}|20\d{2})\b', name)
    if m:
        start = int(m.group(1))
        end_raw = m.group(2)
        end_year = int(end_raw) if len(end_raw) == 4 else int(f"20{end_raw}")
        if end_year == start + 1:
            return f"{start}-{str(end_year)[2:]}", None, "direct"

    # Named season + year: "Fall 2025", "Spring 2026"
    season_m = re.search(r'\b(fall|spring|summer)\s+(20\d{2})\b', name.lower())
    if season_m:
        season_word = season_m.group(1)
        year = int(season_m.group(2))
        if season_word == "fall":
            return f"{year}-{str(year + 1)[2:]}", "Fall", "month_derived"
        elif season_word == "spring":
            return f"{year - 1}-{str(year)[2:]}", "Spring", "month_derived"
        else:
            return f"{year - 1}-{str(year)[2:]}", "Summer", "month_derived"

    # Month name + year: "October 2025", "Aug 2025-March 2026", "_June 2025"
    # Replace underscores with spaces so "_June" becomes " June" and \b triggers correctly
    month_names = "|".join(MONTH_MAP.keys())
    month_m = re.search(
        rf'\b({month_names})\.?\s*(20\d{{2}})\b',
        name.lower().replace("_", " "),
    )
    if month_m:
        month = MONTH_MAP.get(month_m.group(1))
        year = int(month_m.group(2))
        if month is not None:
            if month >= 7:
                ay = f"{year}-{str(year + 1)[2:]}"
                season = "Fall"
            else:
                ay = f"{year - 1}-{str(year)[2:]}"
                season = "Spring"
            return ay, season, "month_derived"

    # Year-only with no month context -- academic year cannot be determined
    if re.search(r'\b20\d{2}\b', name):
        return None, None, "unknown"

    return None, None, "unknown"


def _infer_doc_type(folder_path: str, file_name: str = "") -> str | None:
    # Filename rules take precedence — more specific signal
    file_lower = file_name.lower()
    for doc_type, keywords in DOC_TYPE_FILENAME_RULES:
        if any(kw in file_lower for kw in keywords):
            return doc_type
    # Fall back to folder-path rules
    path_lower = folder_path.lower()
    for doc_type, keywords in DOC_TYPE_RULES:
        if any(kw in path_lower for kw in keywords):
            return doc_type
    return None


_PRECISION_RANK = {"direct": 3, "month_derived": 2, "unknown": 1}


def _find_district(text: str) -> str | None:
    """Return the canonical district name if any known district appears in text."""
    lower = text.lower()
    for district in KNOWN_DISTRICTS:  # ordered longest-first
        if district in lower:
            return DISTRICT_DISPLAY[district]
    return None


def parse_path_metadata(folder_path: str, file_name: str = "") -> dict:
    """
    Parse program, doc_type, academic_year, season, date_precision, district
    from a Drive folder path and file name.

    District and date signals are scanned from both the folder path components
    and the file name so that files like "Lee County_Project Thrive - June 2026.pdf"
    are correctly tagged even when the folder path has no district marker.
    """
    parts = [p.strip() for p in folder_path.split("/") if p.strip()]

    if parts and parts[0].lower() == "data sources":
        parts = parts[1:]

    result: dict = {
        "program": None,
        "doc_type": None,
        "academic_year": None,
        "season": None,
        "date_precision": "unknown",
        "district": None,
    }

    if not parts:
        return result

    # Program: first path component
    program_key = parts[0].lower()
    result["program"] = PROGRAM_MAP.get(program_key)
    if result["program"] is None:
        for key, val in PROGRAM_MAP.items():
            if program_key.startswith(key[:6]):
                result["program"] = val
                break

    # Doc type: second path component
    if len(parts) > 1:
        result["doc_type"] = _infer_doc_type(parts[1], file_name)

    # Scan folder components + filename for date and district signals
    best_precision = "unknown"
    search_targets = parts + ([file_name] if file_name else [])

    for target in search_targets:
        if result["district"] is None:
            result["district"] = _find_district(target)

        ay, season, precision = _parse_academic_year(target)
        if _PRECISION_RANK.get(precision, 0) > _PRECISION_RANK.get(best_precision, 0):
            result["academic_year"] = ay
            result["season"] = season
            result["date_precision"] = precision
            best_precision = precision

    return result

# ---------------------------------------------------------------------------
# File download
# ---------------------------------------------------------------------------

def _file_extension(mime_type: str, original_name: str) -> str:
    if mime_type in GOOGLE_EXPORT_MIME:
        return GOOGLE_EXPORT_MIME[mime_type][1]
    if mime_type in MIME_TO_EXT:
        return MIME_TO_EXT[mime_type]
    suffix = Path(original_name).suffix
    return suffix if suffix else ".bin"


def download_file(
    service, file_id: str, mime_type: str, dest_path: Path
) -> tuple[bool, str | None]:
    """Download or export a single Drive file. Returns (success, error_message)."""
    try:
        if mime_type in GOOGLE_EXPORT_MIME:
            export_mime, _ = GOOGLE_EXPORT_MIME[mime_type]
            content = service.files().export(
                fileId=file_id, mimeType=export_mime
            ).execute()
            dest_path.write_bytes(
                content if isinstance(content, bytes) else content.encode("utf-8")
            )
        else:
            buf = io.BytesIO()
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            dest_path.write_bytes(buf.getvalue())
        return True, None
    except HttpError as e:
        return False, f"HTTP {e.status_code}: {e.reason}"
    except Exception as e:
        return False, str(e)

# ---------------------------------------------------------------------------
# Main fetch logic (used by main() and GoogleDriveSource)
# ---------------------------------------------------------------------------

def fetch_all_documents(service, drive_id: str, dest_dir: Path) -> list[dict]:
    """
    Walk the Shared Drive, download all files into dest_dir, and return
    a list of metadata dicts (same structure as metadata.json records).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching file list...")
    all_items = list_all_items(service, drive_id)
    files = [i for i in all_items if i["mimeType"] != "application/vnd.google-apps.folder"]
    print(f"  {len(files)} files, {len(all_items) - len(files)} folders")

    print("Resolving folder paths...")
    path_map = build_path_map(all_items, drive_id)

    metadata: list[dict] = []
    counts = {"downloaded": 0, "skipped": 0, "exists": 0, "failed": 0}
    now = datetime.now(timezone.utc).isoformat()

    for idx, file in enumerate(files, 1):
        file_id   = file["id"]
        file_name = file["name"]
        mime_type = file["mimeType"]

        if mime_type in SKIP_MIME:
            counts["skipped"] += 1
            continue

        parents     = file.get("parents", [])
        parent_id   = parents[0] if parents else drive_id
        folder_path = path_map.get(parent_id, "")
        parsed      = parse_path_metadata(folder_path, file_name)

        ext        = _file_extension(mime_type, file_name)
        local_path = dest_dir / f"{file_id}{ext}"
        rel_path   = str(local_path.relative_to(PROJECT_ROOT))

        if local_path.exists():
            status, error_msg = "exists", None
            counts["exists"] += 1
        else:
            print(f"  [{idx}/{len(files)}] {file_name[:70]}")
            success, error_msg = download_file(service, file_id, mime_type, local_path)
            if success:
                status = "downloaded"
                counts["downloaded"] += 1
            else:
                status = "error"
                counts["failed"] += 1
                print(f"    x {error_msg}")
            time.sleep(DOWNLOAD_DELAY_SECONDS)

        metadata.append({
            "file_id":         file_id,
            "file_name":       file_name,
            "mime_type":       mime_type,
            "folder_path":     folder_path,
            "local_path":      rel_path,
            "drive_url":       file.get("webViewLink", ""),
            **parsed,
            "download_status": status,
            "error_message":   error_msg,
            "downloaded_at":   now,
        })

    for record in metadata:
        record["file_type"] = FILE_TYPE_MAP.get(record.get("mime_type", ""), "other")
        record["data_form"] = None

    print(f"\n  Downloaded:    {counts['downloaded']}")
    print(f"  Already exist: {counts['exists']}")
    print(f"  Skipped:       {counts['skipped']}")
    print(f"  Failed:        {counts['failed']}")
    return metadata


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    drive_id = os.getenv("SHARED_DRIVE_ID")
    if not drive_id:
        raise EnvironmentError("SHARED_DRIVE_ID is not set in .env")

    print("Connecting to Google Drive...")
    service  = build_drive_service()
    metadata = fetch_all_documents(service, drive_id, DATA_DIR)

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  Metadata:      {METADATA_PATH}")



# ---------------------------------------------------------------------------
# Streaming helpers (no data/raw/ written)
# ---------------------------------------------------------------------------

def fetch_file_bytes(
    service, file_id: str, mime_type: str
) -> tuple["io.BytesIO | None", str, "str | None"]:
    """
    Download a Drive file into memory. Returns (BytesIO, ext, error_or_None).
    The BytesIO is seeked to position 0 on success.
    """
    ext = _file_extension(mime_type, "")
    try:
        if mime_type in GOOGLE_EXPORT_MIME:
            export_mime, ext = GOOGLE_EXPORT_MIME[mime_type]
            raw = service.files().export(
                fileId=file_id, mimeType=export_mime
            ).execute()
            buf = io.BytesIO(raw if isinstance(raw, bytes) else raw.encode("utf-8"))
        else:
            buf = io.BytesIO()
            request = service.files().get_media(fileId=file_id)
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        buf.seek(0)
        return buf, ext, None
    except HttpError as e:
        return None, ext, f"HTTP {e.status_code}: {e.reason}"
    except Exception as e:
        return None, ext, str(e)


def stream_docs(service, drive_id: str):
    """
    Generate (metadata_dict, BytesIO, ext) for every downloadable file in the
    Shared Drive without writing anything to data/raw/.

    Metadata fields match what main() writes to metadata.json.
    """
    print("Connecting to Google Drive...")
    print("Fetching file list...")
    all_items = list_all_items(service, drive_id)
    files = [i for i in all_items if i["mimeType"] != "application/vnd.google-apps.folder"]
    print(f"  {len(files)} files, {len(all_items) - len(files)} folders")

    print("Resolving folder paths...")
    path_map = build_path_map(all_items, drive_id)

    now = datetime.now(timezone.utc).isoformat()
    counts: dict[str, int] = {"streamed": 0, "skipped": 0, "failed": 0}

    for idx, file in enumerate(files, 1):
        file_id   = file["id"]
        file_name = file["name"]
        mime_type = file["mimeType"]

        if mime_type in SKIP_MIME:
            counts["skipped"] += 1
            continue

        parents     = file.get("parents", [])
        parent_id   = parents[0] if parents else drive_id
        folder_path = path_map.get(parent_id, "")
        parsed      = parse_path_metadata(folder_path, file_name)

        ext = _file_extension(mime_type, file_name)
        record = {
            "file_id":         file_id,
            "file_name":       file_name,
            "mime_type":       mime_type,
            "folder_path":     folder_path,
            "local_path":      None,
            "drive_url":       file.get("webViewLink", ""),
            **parsed,
            "file_type":       FILE_TYPE_MAP.get(mime_type, "other"),
            "data_form":       None,
            "download_status": "streamed",
            "error_message":   None,
            "downloaded_at":   now,
        }

        print(f"  [{idx}/{len(files)}] {file_name[:70]}", end="", flush=True)
        buf, ext, error = fetch_file_bytes(service, file_id, mime_type)
        if buf is None:
            record["download_status"] = "error"
            record["error_message"]   = error
            print(f"  x {error}")
            counts["failed"] += 1
            yield record, None, ext
        else:
            print()
            counts["streamed"] += 1
            time.sleep(DOWNLOAD_DELAY_SECONDS)
            yield record, buf, ext

    print(f"\nStreamed: {counts['streamed']}  Skipped: {counts['skipped']}  Failed: {counts['failed']}")
if __name__ == "__main__":
    main()