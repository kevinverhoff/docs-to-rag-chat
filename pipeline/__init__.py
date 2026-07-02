"""
Pipeline orchestrator -- Steps 2 through 6.

Runs each step in sequence. Each step is skipped if its output already exists.

Flags:
  --override       Clear all existing outputs and rerun every step from scratch.
                   By default also deletes data/raw/ so files are re-downloaded.
  --keep-raw       With --override: preserve data/raw/ to skip the re-download.
  --stream         Download and extract without saving raw files to data/raw/.
                   Files are downloaded to memory, extracted, then discarded.

Usage (from project root):
  python pipeline/__init__.py                         # run / resume
  python pipeline/__init__.py --override              # full rerun from scratch
  python pipeline/__init__.py --override --keep-raw   # rerun except re-download
  python pipeline/__init__.py --stream                # no data/raw/ written
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config as _config
DATA_DIR       = PROJECT_ROOT / "data" / "raw"
DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.parquet"
METADATA_PATH  = PROJECT_ROOT / "data" / "metadata.json"
THEMES_RAW     = PROJECT_ROOT / "data" / "themes_raw.parquet"
THEMES         = PROJECT_ROOT / "data" / "themes.parquet"
CHROMA_DIR     = PROJECT_ROOT / "data" / "chroma_db"


def _has_raw_files() -> bool:
    return DATA_DIR.exists() and any(DATA_DIR.iterdir())


def _parquet_has_ok_rows(path: Path, status_col: str) -> bool:
    """Return True only if the parquet exists and has at least one successful row."""
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=[status_col])
        return (df[status_col] == "ok").any()
    except Exception:
        return False


def _chroma_is_built() -> bool:
    return CHROMA_DIR.exists() and any(CHROMA_DIR.iterdir())


def _clear_outputs(keep_raw: bool = False) -> None:
    """Delete pipeline outputs so every step is forced to rerun."""
    if not keep_raw and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"  removed data/raw/")
    if METADATA_PATH.exists():
        METADATA_PATH.unlink()
        print(f"  removed metadata.json")
    for path in [DOCUMENTS_PATH, THEMES_RAW, THEMES]:
        if path.exists():
            path.unlink()
            print(f"  removed {path.name}")
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
        print(f"  removed chroma_db/")


def main(override: bool = False, keep_raw: bool = False, stream: bool = False) -> None:

    if override:
        print("=== --override: clearing existing pipeline outputs ===")
        _clear_outputs(keep_raw=keep_raw)
        print()

    # ------------------------------------------------------------------
    # Steps 2-3: Fetch documents and extract text
    # ------------------------------------------------------------------
    if _config.SOURCE_TYPE == "local":
        if METADATA_PATH.exists() and not override:
            print("[skip] Step 2  -- metadata.json already exists")
        else:
            print("=== Step 2: Indexing local document folder ===")
            from pipeline.sources.local import LocalFolderSource
            if not _config.LOCAL_DOCS_DIR:
                raise EnvironmentError(
                    "LOCAL_DOCS_DIR must be set in .env when SOURCE_TYPE=local"
                )
            source  = LocalFolderSource(Path(_config.LOCAL_DOCS_DIR))
            records = source.fetch_documents(DATA_DIR)
            METADATA_PATH.write_text(
                json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(f"  Indexed {len(records)} files from {_config.LOCAL_DOCS_DIR}")

        if DOCUMENTS_PATH.exists():
            print("[skip] Step 3  -- documents.parquet already exists")
        else:
            print("=== Step 3: Extracting text ===")
            from ingest import main as run_ingest
            run_ingest()

    elif stream and not DOCUMENTS_PATH.exists():
        print("=== Steps 2-3 (stream): downloading and extracting without data/raw/ ===")
        drive_id = os.getenv("SHARED_DRIVE_ID")
        if not drive_id:
            raise EnvironmentError("SHARED_DRIVE_ID is not set in .env")

        from get_docs import build_drive_service, stream_docs as _stream_docs
        from ingest import process_stream

        service = build_drive_service()
        meta_records: list[dict] = []

        def _collecting_gen():
            for rec, buf, ext in _stream_docs(service, drive_id):
                meta_records.append(rec)
                yield rec, buf, ext

        process_stream(_collecting_gen())
        METADATA_PATH.write_text(
            json.dumps(meta_records, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    elif stream and DOCUMENTS_PATH.exists():
        print(f"[skip] Steps 2-3 -- documents.parquet already exists")

    else:
        # ------------------------------------------------------------------
        # Step 2: Download documents from Google Drive
        # ------------------------------------------------------------------
        if _has_raw_files():
            file_count = sum(1 for _ in DATA_DIR.iterdir())
            print(f"[skip] Step 2  -- data/raw/ already has {file_count} files")
        else:
            print("=== Step 2: Downloading documents ===")
            from get_docs import main as run_get_docs
            run_get_docs()

        # ------------------------------------------------------------------
        # Step 3: Extract text from all downloaded files
        # ------------------------------------------------------------------
        if DOCUMENTS_PATH.exists():
            print(f"[skip] Step 3  -- documents.parquet already exists")
        else:
            print("=== Step 3: Extracting text ===")
            from ingest import main as run_ingest
            run_ingest()

    # ------------------------------------------------------------------
    # Step 3.5: LLM theme extraction (one call per document)
    # ------------------------------------------------------------------
    if _parquet_has_ok_rows(THEMES_RAW, "theme_extraction_status"):
        ok_count = pd.read_parquet(THEMES_RAW, columns=["theme_extraction_status"])
        ok_count = (ok_count["theme_extraction_status"] == "ok").sum()
        print(f"[skip] Step 3.5 -- themes_raw.parquet has {ok_count} extracted documents")
    else:
        print("=== Step 3.5: Extracting themes ===")
        from extract_themes import main as run_extract_themes
        run_extract_themes()

    # ------------------------------------------------------------------
    # Step 3.6: Canonicalise + cluster themes
    # ------------------------------------------------------------------
    if _parquet_has_ok_rows(THEMES, "theme_extraction_status"):
        ok_count = pd.read_parquet(THEMES, columns=["theme_extraction_status"])
        ok_count = (ok_count["theme_extraction_status"] == "ok").sum()
        print(f"[skip] Step 3.6 -- themes.parquet has {ok_count} documents")
    else:
        print("=== Step 3.6: Deduplicating themes ===")
        from deduplicate_themes import main as run_deduplicate
        run_deduplicate()

    # ------------------------------------------------------------------
    # Steps 4-6: Chunk, embed, and index into Chroma
    # ------------------------------------------------------------------
    if _chroma_is_built():
        print(f"[skip] Steps 4-6 -- chroma_db/ already exists")
    else:
        print("=== Steps 4-6: Building vector store ===")
        from build_vectorstore import main as run_vectorstore
        run_vectorstore(rebuild=override)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the document ingestion pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python pipeline/__init__.py                         # run / resume
  python pipeline/__init__.py --override              # full rerun from scratch
  python pipeline/__init__.py --override --keep-raw   # rerun except re-download
  python pipeline/__init__.py --stream                # no data/raw/ written
""",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Delete all existing outputs and rerun every step.",
    )
    parser.add_argument(
        "--keep-raw",
        action="store_true",
        dest="keep_raw",
        help="With --override: preserve data/raw/ to skip re-downloading from Google Drive.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Download and extract without saving raw files to data/raw/.",
    )
    args = parser.parse_args()
    main(override=args.override, keep_raw=args.keep_raw, stream=args.stream)