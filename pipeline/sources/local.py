"""
Local folder document source.

Walks a directory tree and returns a DocumentRecord for each supported file.
No download is performed -- local_path points to the file directly.
drive_url is always empty string.
file_id is a stable SHA-1 hash of the file path relative to root.

Folder structure convention (optional):
  <root>/<program>/<doc_type>/<file>   ->  program = depth-0 folder name
                                           doc_type = depth-1 folder name
"""
from __future__ import annotations

import hashlib
import mimetypes
from pathlib import Path

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
        """
        records: list[DocumentRecord] = []

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() in SKIP_EXTENSIONS:
                continue

            rel      = path.relative_to(self.root)
            file_id  = hashlib.sha1(str(rel).encode()).hexdigest()[:16]
            mime, _  = mimetypes.guess_type(str(path))
            parts    = rel.parts  # (program, doc_type?, ..., filename)

            records.append(DocumentRecord(
                file_id        = file_id,
                file_name      = path.name,
                mime_type      = mime or "application/octet-stream",
                folder_path    = str(rel.parent) if len(parts) > 1 else "",
                local_path     = str(path),
                drive_url      = "",
                program        = parts[0] if len(parts) > 1 else None,
                doc_type       = parts[1] if len(parts) > 2 else None,
                academic_year  = None,
                season         = None,
                date_precision = "unknown",
                district       = None,
                download_status= "exists",
            ))

        return records
