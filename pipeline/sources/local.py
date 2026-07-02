"""
Local folder document source.

Walks a directory tree and returns a DocumentRecord for each supported file.
No download is performed -- local_path points to the file directly.
drive_url is always empty string.
file_id is a stable SHA-1 hash of the file path relative to root.

Folder structure -> tags via FOLDER_METADATA_LEVELS config:
  FOLDER_METADATA_LEVELS=department,category
  root/HR/Reports/file.pdf  ->  tags = {"department": "HR", "category": "Reports"}
"""
from __future__ import annotations

import hashlib
import mimetypes
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as _config
from pipeline.sources.base import DocumentRecord

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp",
    ".mp4", ".mp3", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".rar",
    ".bin", ".exe", ".dll",
}


class LocalFolderSource:
    """Document source that reads files from a local directory tree."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()

    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]:
        """
        Walk self.root, returning one DocumentRecord per supported file.
        dest_dir is accepted for interface compatibility but not used --
        files are already local.

        Tag names come from FOLDER_METADATA_LEVELS in config. Folder depth 0
        maps to the first name, depth 1 to the second, and so on. Levels beyond
        what is configured are silently ignored.
        """
        level_names = _config.FOLDER_METADATA_LEVELS
        records: list[DocumentRecord] = []

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue

            rel     = path.relative_to(self.root)
            file_id = hashlib.sha1(str(rel).encode()).hexdigest()[:16]
            mime, _ = mimetypes.guess_type(str(path))
            parts   = rel.parts  # (folder0, folder1, ..., filename)

            tags: dict[str, str] = {}
            for i, name in enumerate(level_names):
                if i < len(parts) - 1:
                    tags[name] = parts[i]

            records.append(DocumentRecord(
                file_id         = file_id,
                file_name       = path.name,
                mime_type       = mime or "application/octet-stream",
                folder_path     = str(rel.parent) if len(parts) > 1 else "",
                local_path      = str(path),
                drive_url       = "",
                download_status = "exists",
                tags            = tags,
            ))

        return records