"""
Google Drive document source.

Delegates to get_docs.fetch_all_documents() so Drive API logic is not duplicated.
Fields parsed from the Drive folder structure (program, doc_type, academic_year,
season, district, etc.) are preserved as tags so existing pipelines continue to work.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from pipeline.sources.base import DocumentRecord

# Fields produced by get_docs.fetch_all_documents() that become tags.
# Any non-null, non-empty value is included.
_TAG_FIELDS = [
    "program", "doc_type", "academic_year", "season",
    "date_precision", "district",
]


class GoogleDriveSource:
    """Fetch documents from a Google Shared Drive."""

    def __init__(self, drive_id: str, creds_path: Path | None = None) -> None:
        self.drive_id   = drive_id
        self.creds_path = creds_path

    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]:
        """
        Download all files from the Shared Drive into dest_dir and return
        their metadata as DocumentRecord dicts.
        """
        import get_docs

        if self.drive_id:
            os.environ["SHARED_DRIVE_ID"] = self.drive_id

        service     = get_docs.build_drive_service()
        raw_records = get_docs.fetch_all_documents(service, self.drive_id, dest_dir)

        result: list[DocumentRecord] = []
        for rec in raw_records:
            tags = {
                k: str(v)
                for k in _TAG_FIELDS
                if (v := rec.get(k)) is not None and str(v).strip()
            }
            result.append(DocumentRecord(
                file_id         = rec.get("file_id", ""),
                file_name       = rec.get("file_name", ""),
                mime_type       = rec.get("mime_type", ""),
                folder_path     = rec.get("folder_path", ""),
                local_path      = rec.get("local_path", ""),
                drive_url       = rec.get("drive_url", ""),
                download_status = rec.get("download_status", ""),
                tags            = tags,
            ))
        return result