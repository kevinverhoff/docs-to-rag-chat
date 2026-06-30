"""
Google Drive document source.

Delegates to get_docs.fetch_all_documents() so Drive API logic is not duplicated.
Returns the same metadata structure that get_docs.main() writes to metadata.json,
as a list of DocumentRecord dicts.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_PIPELINE_DIR = Path(__file__).parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))

from pipeline.sources.base import DocumentRecord


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

        service = get_docs.build_drive_service()
        return get_docs.fetch_all_documents(service, self.drive_id, dest_dir)
