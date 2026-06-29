"""
Step 7: RAG Pipeline

RagPipeline wraps Chroma + OpenAI into a single cached object.
Streamlit initializes it once with @st.cache_resource; every query calls
pipeline.answer(question, **filters) on the same instance.

Usage:
  from rag_pipeline import RagPipeline
  pipeline = RagPipeline()
  result = pipeline.answer(
      "What are the main challenges in SWS implementation?",
      program="SWS", district="Lake",
  )
  print(result["answer"])
  for src in result["sources"]:
      print(f"[{src[''n'']}] {src[''file_name'']}  {src[''drive_url'']}")
"""

import json
from pathlib import Path

import chromadb
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")

CHROMA_PATH = PROJECT_ROOT / "data" / "chroma_db"
THEMES_PATH = PROJECT_ROOT / "data" / "themes.parquet"
COLLECTION  = "docs"
EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
TEMPERATURE = 0.1

TOP_K       = 20  # over-fetch before dedup
MAX_PER_DOC = 2   # max chunks from the same document
FINAL_TOP_K = 8   # chunks passed to the LLM

# Keywords that signal the user wants cross-document theme/trend analysis
THEME_KEYWORDS = {
    "theme", "themes", "trend", "trends", "pattern", "patterns",
    "common", "across", "compare", "comparison",
    "over time", "finding", "findings",
    "cluster", "clusters", "prevalence", "frequency",
    "most common", "least common", "what has changed", "how has",
}

SYSTEM_PROMPT = (PROJECT_ROOT / "prompts" / "rag_pipeline_system_prompt.txt").read_text(encoding="utf-8")


class RagPipeline:
    """
    Owns the Chroma and OpenAI connections. Designed to be initialized once
    via @st.cache_resource in Streamlit and reused across queries.
    """

    def __init__(
        self,
        chroma_path: Path = CHROMA_PATH,
        themes_path: Path = THEMES_PATH,
    ) -> None:
        self.openai = OpenAI()

        chroma = chromadb.PersistentClient(path=str(chroma_path))
        try:
            self.collection = chroma.get_collection(COLLECTION)
        except Exception:
            raise RuntimeError(
                f"Chroma collection '{COLLECTION}' not found. "
                "Run pipeline/build_vectorstore.py first."
            )

        self.themes_df: pd.DataFrame | None = None
        if themes_path.exists():
            self.themes_df = pd.read_parquet(
                themes_path,
                columns=["file_id", "themes", "key_findings", "theme_clusters"],
            )

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def answer(
        self,
        question: str,
        *,
        program: str | None = None,
        district: str | None = None,
        academic_year: str | None = None,
        doc_type: str | None = None,
        theme_cluster: str | None = None,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Run a full RAG query.

        Returns:
          {
            "answer":  str,
            "sources": [
              {
                "n", "file_name", "drive_url",
                "program", "district", "academic_year", "doc_type",
                "section", "text"
              }, ...
            ]
          }
        """
        where  = _build_where(program, district, academic_year, doc_type, theme_cluster)
        chunks = self._retrieve(question, where)

        if not chunks:
            suffix = " with the current filters applied" if where else ""
            return {
                "answer": f"I couldn't find any relevant documents for that question{suffix}.",
                "sources": [],
            }

        theme_ctx = ""
        if self.themes_df is not None and _is_theme_question(question):
            theme_ctx = self._theme_context(chunks)

        messages = _build_messages(question, chunks, theme_ctx, history)

        response = self.openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=TEMPERATURE,
        )
        answer_text = response.choices[0].message.content.strip()

        sources = [
            {
                "n":            c["n"],
                "file_name":    c["meta"].get("file_name", ""),
                "drive_url":    c["meta"].get("drive_url", ""),
                "program":      c["meta"].get("program", ""),
                "district":     c["meta"].get("district", ""),
                "academic_year":c["meta"].get("academic_year", ""),
                "doc_type":     c["meta"].get("doc_type", ""),
                "section":      _section_label(c["meta"]),
                "text":         c["text"],
            }
            for c in chunks
        ]

        return {"answer": answer_text, "sources": sources}

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _retrieve(self, question: str, where: dict | None) -> list[dict]:
        """Embed the question, query Chroma, dedupe to MAX_PER_DOC per doc."""
        emb = self.openai.embeddings.create(model=EMBED_MODEL, input=[question])
        query_vec = emb.data[0].embedding

        try:
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=TOP_K,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception:
            return []

        docs   = results["documents"][0]
        metas  = results["metadatas"][0]
        dists  = results["distances"][0]

        # Keep best-scoring chunks; at most MAX_PER_DOC per file_id
        seen: dict[str, int] = {}
        pruned: list[tuple] = []
        for text, meta, dist in zip(docs, metas, dists):
            fid = meta.get("file_id", "")
            if seen.get(fid, 0) < MAX_PER_DOC:
                seen[fid] = seen.get(fid, 0) + 1
                pruned.append((text, meta, dist))
            if len(pruned) >= FINAL_TOP_K:
                break

        return [
            {"n": i + 1, "text": text, "meta": meta, "distance": dist}
            for i, (text, meta, dist) in enumerate(pruned)
        ]

    def _theme_context(self, chunks: list[dict]) -> str:
        """Build a theme-analysis block for the retrieved documents."""
        file_ids = list({c["meta"].get("file_id") for c in chunks if c["meta"].get("file_id")})
        rows = self.themes_df[self.themes_df["file_id"].isin(file_ids)]
        if rows.empty:
            return ""

        # Map file_id -> file_name from chunk metadata
        id_to_name = {
            c["meta"].get("file_id"): c["meta"].get("file_name", "")
            for c in chunks
        }

        lines = ["\nTHEME ANALYSIS FOR RETRIEVED DOCUMENTS:"]
        for _, row in rows.iterrows():
            fid     = row["file_id"]
            fname   = id_to_name.get(fid, fid)
            clusters = _parse_json_list(row.get("theme_clusters"))
            themes   = _parse_json_list(row.get("themes"))
            findings = _parse_json_list(row.get("key_findings"))

            lines.append(f"\n{fname}")
            if clusters:
                lines.append(f"  Clusters: {', '.join(clusters)}")
            if themes:
                lines.append(f"  Themes:   {', '.join(themes)}")
            for finding in findings:
                lines.append(f"  - {finding}")

        return "\n".join(lines)


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_where(
    program: str | None,
    district: str | None,
    academic_year: str | None,
    doc_type: str | None,
    theme_cluster: str | None,
) -> dict | None:
    conditions = []
    if program:       conditions.append({"program":       {"$eq":       program}})
    if district:      conditions.append({"district":      {"$eq":       district}})
    if academic_year: conditions.append({"academic_year": {"$eq":       academic_year}})
    if doc_type:      conditions.append({"doc_type":      {"$eq":       doc_type}})
    if theme_cluster: conditions.append({"theme_clusters":{"$contains": theme_cluster}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _is_theme_question(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in THEME_KEYWORDS)


def _section_label(meta: dict) -> str:
    parts = [meta.get(f"section_h{i}") for i in [1, 2, 3]]
    return " > ".join(p for p in parts if p)


def _parse_json_list(val) -> list[str]:
    if not val:
        return []
    try:
        parsed = json.loads(val) if isinstance(val, str) else val
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def _build_messages(
    question: str,
    chunks: list[dict],
    theme_context: str,
    history: list[dict] | None,
) -> list[dict]:
    blocks = []
    for c in chunks:
        meta = c["meta"]
        header = []
        if meta.get("program"):       header.append(f"Program: {meta['program']}")
        if meta.get("district"):      header.append(f"District: {meta['district']}")
        if meta.get("academic_year"): header.append(f"Year: {meta['academic_year']}")
        if meta.get("season"):        header.append(f"Season: {meta['season']}")
        section = _section_label(meta)

        block = f"[{c['n']}] {meta.get('file_name', '')}"
        if header:
            block += f"\n    {' | '.join(header)}"
        if section:
            block += f"\n    Section: {section}"
        block += f"\n    ---\n    {c['text']}"
        blocks.append(block)

    context = "\n\n".join(blocks)
    if theme_context:
        context += "\n" + theme_context

    user_content = f"CONTEXT DOCUMENTS:\n\n{context}\n\nQUESTION: {question}"

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_content})
    return messages

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask a question against the document library.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python rag_pipeline.py "What challenges have districts reported with SWS?"
  python rag_pipeline.py "What are teachers saying?" --district Lake --program SWS
  python rag_pipeline.py "How has teacher buy-in changed?" --year 2024-25 --theme-cluster "Educator Development"
""",
    )

    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--program",  default=None, help="Filter by program (e.g. SWS, Focus K-3)")
    parser.add_argument("--district", default=None, help="Filter by district (e.g. Lake, Osceola)")
    parser.add_argument("--year",     default=None, help="Filter by academic year (e.g. 2024-25)")
    parser.add_argument("--doc-type", default=None, dest="doc_type", help="Filter by doc type (e.g. site_visit)")
    parser.add_argument("--theme-cluster", default=None, dest="theme_cluster",
                        help="Filter by high-level theme cluster")

    args = parser.parse_args()

    pipeline = RagPipeline()
    result = pipeline.answer(
        args.question,
        program=args.program,
        district=args.district,
        academic_year=args.year,
        doc_type=args.doc_type,
        theme_cluster=args.theme_cluster,
    )

    print(result["answer"])

    if result["sources"]:
        print()
        print("Sources:")
        for s in result["sources"]:
            line = f"  [{s['n']}] {s['file_name']}"
            if s["district"]:      line += f"  |  {s['district']}"
            if s["academic_year"]: line += f"  |  {s['academic_year']}"
            print(line)
            if s["drive_url"]:
                print(f"       {s['drive_url']}")