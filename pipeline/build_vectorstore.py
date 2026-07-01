"""
Steps 4-6: Chunker + Embedder + Vector Store

Reads documents.parquet, chunks each document (header-aware for DOCX files),
embeds with text-embedding-3-small, and indexes into a local Chroma collection
with filterable metadata for every structured field.

Usage:
  python pipeline/build_vectorstore.py
  python pipeline/build_vectorstore.py --rebuild   # drop and rebuild from scratch
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / "secrets" / ".env")

import config as _config
from providers import build_embedder

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.parquet"
CHROMA_PATH    = PROJECT_ROOT / "data" / "chroma_db"
THEMES_PATH    = PROJECT_ROOT / "data" / "themes.parquet"
COLLECTION     = _config.CHROMA_COLLECTION

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
CHUNK_MIN     = 100
EMBED_BATCH   = 100
REQUEST_DELAY = 0.05  # seconds between embedding batch calls


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """
    Recursively split text at natural boundaries.
    Priority: paragraph breaks -> line breaks -> sentence ends -> word spaces.
    """
    if len(text) <= CHUNK_SIZE:
        return [text] if len(text.strip()) >= CHUNK_MIN else []

    for sep in ("\n\n", "\n", ". ", " "):
        if sep not in text:
            continue

        pieces = text.split(sep)
        chunks: list[str] = []
        buf = ""

        for piece in pieces:
            candidate = (buf + sep + piece).lstrip(sep) if buf else piece
            if len(candidate) <= CHUNK_SIZE:
                buf = candidate
            elif buf:
                chunks.append(buf.strip())
                tail = buf[-CHUNK_OVERLAP:].lstrip() if len(buf) > CHUNK_OVERLAP else buf
                buf = (tail + sep + piece).lstrip(sep) if tail else piece
            else:
                chunks.append(piece.strip())
                buf = ""

        if buf and len(buf.strip()) >= CHUNK_MIN:
            chunks.append(buf.strip())

        result = [c for c in chunks if len(c) >= CHUNK_MIN]
        if result:
            return result

    # Last resort: hard split with overlap
    out = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i : i + CHUNK_SIZE].strip()
        if len(chunk) >= CHUNK_MIN:
            out.append(chunk)
    return out


def chunk_document(rec: dict) -> list[dict]:
    """
    Produce a list of chunk dicts from one documents.parquet row.

    For DOCX files with heading data, splits at heading boundaries first
    so chunks never span section headers. All other formats split directly.

    Each chunk dict:
      text, embed_text, section_h1, section_h2, section_h3, chunk_index, chunk_count
    """
    text = rec.get("text") or ""
    if not text.strip():
        return []

    # (chunk_text, h1, h2, h3)
    raw_chunks: list[tuple[str, str | None, str | None, str | None]] = []

    headings: list[dict] = []
    if rec.get("local_path", "").endswith(".docx"):
        raw = rec.get("headings")
        if raw:
            try:
                headings = json.loads(raw) if isinstance(raw, str) else raw
                headings = sorted(headings, key=lambda h: h.get("char_offset", 0))
            except (json.JSONDecodeError, TypeError):
                headings = []

    if headings:
        # Chunk text before the first heading
        pre = text[: headings[0]["char_offset"]].strip()
        if pre:
            for chunk in _split_text(pre):
                raw_chunks.append((chunk, None, None, None))

        # Chunk each section (heading boundary to next heading boundary)
        offsets = [h["char_offset"] for h in headings] + [len(text)]
        current_h: dict[int, str | None] = {1: None, 2: None, 3: None}

        for i, heading in enumerate(headings):
            level = heading.get("level", 1)
            current_h[level] = heading.get("text") or None
            for lower in range(level + 1, 4):
                current_h[lower] = None

            section = text[offsets[i] : offsets[i + 1]].strip()
            if not section:
                continue
            for chunk in _split_text(section):
                raw_chunks.append((chunk, current_h[1], current_h[2], current_h[3]))
    else:
        for chunk in _split_text(text):
            raw_chunks.append((chunk, None, None, None))

    chunk_count = len(raw_chunks)
    result = []
    for idx, (chunk_text, h1, h2, h3) in enumerate(raw_chunks):
        section_label = " > ".join(h for h in [h1, h2, h3] if h) or None
        prefix = _build_prefix(rec, section_label)
        result.append({
            "text":        chunk_text,
            "embed_text":  f"{prefix}\n\n{chunk_text}",
            "section_h1":  h1,
            "section_h2":  h2,
            "section_h3":  h3,
            "chunk_index": idx,
            "chunk_count": chunk_count,
        })
    return result


def _build_prefix(rec: dict, section: str | None) -> str:
    """Contextual header prepended to each chunk before embedding (improves retrieval)."""
    lines = [f"File: {rec.get('file_name', '')}"]
    if rec.get("folder_path"):
        lines.append(f"Folder: {rec['folder_path']}")
    try:
        tags = json.loads(rec.get("tags") or "{}")
    except (json.JSONDecodeError, TypeError):
        tags = {}
    parts = [
        f"{k.replace('_', ' ').title()}: {v}"
        for k, v in tags.items() if v
    ]
    if parts:
        lines.append(" | ".join(parts))
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Embedding + indexing
# ---------------------------------------------------------------------------

def _safe(val) -> str | int | float | bool:
    """Chroma metadata requires no None values."""
    if val is None:
        return ""
    if isinstance(val, (int, float, bool)):
        return val
    return str(val)


def embed_and_index(
    embedder,
    collection: chromadb.Collection,
    chunks: list[dict],
    rec: dict,
) -> int:
    """Embed all chunks for one document and upsert into Chroma. Returns chunk count."""
    if not chunks:
        return 0

    embed_texts    = [c["embed_text"] for c in chunks]
    all_embeddings = embedder.embed_documents(embed_texts)

    ids       = [f"{rec['file_id']}_chunk_{c['chunk_index']}" for c in chunks]
    documents = [c["text"] for c in chunks]

    try:
        tags: dict = json.loads(rec.get("tags") or "{}")
    except (json.JSONDecodeError, TypeError):
        tags = {}

    metadatas = [
        {
            "file_id":        _safe(rec.get("file_id")),
            "file_name":      _safe(rec.get("file_name")),
            "drive_url":      _safe(rec.get("drive_url")),
            "folder_path":    _safe(rec.get("folder_path")),
            **{k: _safe(v) for k, v in tags.items()},
            "section_h1":     _safe(c.get("section_h1")),
            "section_h2":     _safe(c.get("section_h2")),
            "section_h3":     _safe(c.get("section_h3")),
            "chunk_index":    c["chunk_index"],
            "chunk_count":    c["chunk_count"],
            "theme_clusters": _safe(rec.get("theme_clusters")),
        }
        for c in chunks
    ]

    collection.upsert(
        ids=ids,
        embeddings=all_embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(rebuild: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector store")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and rebuild the collection from scratch")
    args, _ = parser.parse_known_args()
    do_rebuild = rebuild or args.rebuild

    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError("documents.parquet not found -- run ingest.py first")

    df = pd.read_parquet(DOCUMENTS_PATH)
    processable = df[df["extraction_status"] == "ok"]
    print(f"Documents available: {len(processable)} / {len(df)} total")

    # Build file_id -> pipe-separated cluster string from themes.parquet if present
    cluster_lookup: dict[str, str] = {}
    if THEMES_PATH.exists():
        themes_df = pd.read_parquet(THEMES_PATH, columns=["file_id", "theme_clusters"])
        for _, trow in themes_df.iterrows():
            raw = trow.get("theme_clusters")
            if raw:
                try:
                    clusters = json.loads(raw) if isinstance(raw, str) else raw
                    cluster_lookup[str(trow["file_id"])] = " | ".join(clusters)
                except (json.JSONDecodeError, TypeError):
                    pass
        print(f"Theme clusters loaded for {len(cluster_lookup)} / {len(themes_df)} documents")
    else:
        print("themes.parquet not found -- theme_clusters will be empty in metadata")

    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))

    if do_rebuild:
        try:
            chroma.delete_collection(COLLECTION)
            print("Dropped existing collection.")
        except Exception:
            pass

    collection = chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Resolve already-indexed file_ids for idempotent resume
    already_indexed: set[str] = set()
    existing_count = collection.count()
    if existing_count > 0 and not do_rebuild:
        result = collection.get(include=["metadatas"])
        already_indexed = {m["file_id"] for m in result["metadatas"] if m.get("file_id")}
        print(f"Resuming: {len(already_indexed)} files already indexed ({existing_count} chunks)")

    to_process = processable[~processable["file_id"].isin(already_indexed)]
    print(f"Files to index: {len(to_process)}")

    if len(to_process) == 0:
        print("Nothing to do.")
        return

    embedder     = build_embedder()
    total_chunks = 0
    errors = 0

    for idx, (_, record) in enumerate(to_process.iterrows(), 1):
        rec = record.to_dict()
        label = rec.get("file_name", "")[:50]

        try:
            rec["theme_clusters"] = cluster_lookup.get(str(rec.get("file_id", "")), "")
            chunks = chunk_document(rec)
            if not chunks:
                print(f"  [{idx}/{len(to_process)}] SKIP (no chunks)  {label}")
                continue

            added = embed_and_index(embedder, collection, chunks, rec)
            total_chunks += added
            print(f"  [{idx}/{len(to_process)}] OK  {added:>3} chunks  {label}")

        except Exception:
            errors += 1
            last_line = traceback.format_exc().strip().splitlines()[-1]
            print(f"  [{idx}/{len(to_process)}] ERROR  {label}")
            print(f"    {last_line}")
            if "insufficient_quota" in last_line:
                print("\nQuota exhausted -- add credits at https://platform.openai.com/settings/billing")
                print("Re-run after adding credits; already-indexed files will be skipped.")
                break

    print(f"\nDone.")
    print(f"  Chunks added this run: {total_chunks}")
    print(f"  Errors:                {errors}")
    print(f"  Total chunks in DB:    {collection.count()}")
    print(f"  Store path:            {CHROMA_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Chroma vector store")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and rebuild the collection from scratch")
    _args = parser.parse_args()
    main(rebuild=_args.rebuild)
