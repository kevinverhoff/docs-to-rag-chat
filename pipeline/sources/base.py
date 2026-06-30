"""
DocumentSource protocol and DocumentRecord TypedDict.

Both pipeline/sources/gdrive.py and pipeline/sources/local.py must produce
list[DocumentRecord] from their fetch_documents() method. The rest of the
pipeline (ingest.py, build_vectorstore.py) reads these fields unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class DocumentRecord(TypedDict):
    file_id:         str
    file_name:       str
    mime_type:       str
    folder_path:     str
    local_path:      str          # absolute path to the downloaded/existing file
    drive_url:       str          # empty string for local source
    program:         str | None
    doc_type:        str | None
    academic_year:   str | None
    season:          str | None
    date_precision:  str          # "direct" | "month_derived" | "unknown"
    district:        str | None
    download_status: str          # "downloaded" | "exists" | "error" | "skipped"


class DocumentSource(Protocol):
    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]:
        """
        Populate dest_dir with files (if needed) and return one DocumentRecord
        per document. dest_dir will be created if it does not exist.
        """
        ...
