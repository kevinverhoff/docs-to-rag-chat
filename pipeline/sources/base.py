from __future__ import annotations
from pathlib import Path
from typing import Protocol, TypedDict


class DocumentRecord(TypedDict):
    file_id: str
    file_name: str
    mime_type: str
    folder_path: str
    local_path: str
    drive_url: str
    download_status: str
    tags: dict[str, str]


class DocumentSource(Protocol):
    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]: ...