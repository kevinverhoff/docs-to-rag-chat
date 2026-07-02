"""
Step 8: Tool definitions for the document agent.

Tools are created via make_tools(pipeline, themes_df) which returns a list
of LangChain tools with the pipeline and themes dataframe in closure scope.

Q&A tools (vector store):
  search         -- find relevant documents by topic
  answer         -- full RAG answer with inline citations
  summarize      -- broader synthesis across more documents
  extract_quotes -- verbatim passages with source attribution

Cross-dimensional tools (themes.parquet):
  browse_themes  -- filter the themes table by tag filters and theme cluster
  compare        -- how a topic appears across a given dimension
  synthesize     -- synthesize results from the most recent compare() call
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.messages import HumanMessage as _HumanMessage, SystemMessage as _SystemMessage
from langchain_core.tools import tool

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from providers import build_chat_model as _build_chat_model
from rag_pipeline import RagPipeline, _build_where

PROJECT_ROOT = Path(__file__).parent

# Spreadsheet MIME types used to identify survey files
_SPREADSHEET_MIMES = {
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/vnd.ms-excel.sheet.macroenabled.12",
    "text/csv",
}


def make_tools(
    pipeline: RagPipeline,
    themes_df: pd.DataFrame | None,
    metadata: list[dict] | None = None,
) -> list:
    """
    Returns all agent tools with pipeline and themes_df captured in closure.
    Pass the result directly to create_react_agent().
    """

    _synth_llm = _build_chat_model()

    # Verified file_name -> drive_url mapping from themes.parquet.
    # Overrides whatever URL the vector store chunk metadata contains.
    url_lookup: dict[str, str] = {}
    if themes_df is not None:
        for _, _row in themes_df.dropna(subset=["drive_url"]).iterrows():
            _fn = _row.get("file_name", "")
            _url = _row.get("drive_url", "")
            if _fn and _url and _fn not in url_lookup:
                url_lookup[_fn] = _url

    # ------------------------------------------------------------------
    # Q&A tools
    # ------------------------------------------------------------------

    @tool
    def search(
        query: str,
        tag_filters: Optional[dict] = None,
        theme_cluster: Optional[str] = None,
    ) -> str:
        """
        Search for documents relevant to a query. Returns document titles,
        tag metadata, and similarity scores.
        Use this to discover what documents exist on a topic before calling answer().
        tag_filters: optional dict of tag key/value pairs, e.g. {"program": "Ekumen Outreach"}.
        theme_cluster: optional theme cluster string to restrict results.
        """
        where = _build_where(tag_filters, theme_cluster)
        chunks = pipeline._retrieve(query, where)
        if not chunks:
            return "No relevant documents found."

        _META_SKIP = {
            "file_id", "file_name", "drive_url", "folder_path",
            "section_h1", "section_h2", "section_h3",
            "chunk_index", "chunk_count", "theme_clusters",
        }
        seen: set[str] = set()
        lines = []
        for c in chunks:
            fid = c["meta"].get("file_id", "")
            if fid in seen:
                continue
            seen.add(fid)
            meta = c["meta"]
            score = round(1 - c["distance"], 3)
            parts = [meta.get("file_name", "unknown")]
            for k, v in meta.items():
                if k not in _META_SKIP and v:
                    parts.append(f"{k}={v}")
            parts.append(f"score={score}")
            lines.append("- " + "  |  ".join(parts))

        return "\n".join(lines)

    @tool
    def answer(
        query: str,
        tag_filters: Optional[dict] = None,
        theme_cluster: Optional[str] = None,
    ) -> str:
        """
        Answer a specific question using retrieved document passages. Returns a
        cited answer with inline [1], [2] references and a Sources block with
        Google Drive links. Use for direct factual questions about documents.
        tag_filters: optional dict of tag key/value pairs to narrow retrieval.
        theme_cluster: optional theme cluster string to restrict results.
        """
        result = pipeline.answer(
            query,
            tag_filters=tag_filters,
            theme_cluster=theme_cluster,
        )
        return _format_rag_result(result, url_lookup)

    @tool
    def summarize(
        query: str,
        tag_filters: Optional[dict] = None,
    ) -> str:
        """
        Synthesize a high-level overview across multiple documents on a topic.
        Retrieves more documents than answer() for a broader picture.
        Use when the user wants general understanding rather than a specific fact.
        tag_filters: optional dict of tag key/value pairs to narrow retrieval.
        """
        synthesis_q = f"Synthesize an overview and key takeaways about: {query}"
        result = pipeline.answer(
            synthesis_q,
            tag_filters=tag_filters,
        )
        return _format_rag_result(result, url_lookup)

    @tool
    def extract_quotes(
        query: str,
        tag_filters: Optional[dict] = None,
    ) -> str:
        """
        Return verbatim passages from documents relevant to a query.
        Each passage is attributed to its source document with a Google Drive link.
        Use when the user explicitly wants direct quotes or exact language.
        tag_filters: optional dict of tag key/value pairs to narrow retrieval.
        """
        where = _build_where(tag_filters, None)
        chunks = pipeline._retrieve(query, where)
        if not chunks:
            return "No relevant documents found."

        lines = []
        for c in chunks:
            meta  = c["meta"]
            fname = meta.get("file_name", "")
            url   = url_lookup.get(fname, "") or meta.get("drive_url", "")
            link  = f"[{fname}]({url})" if url else fname
            attr_parts = [link]
            if meta.get("district"):      attr_parts.append(meta["district"])
            if meta.get("academic_year"): attr_parts.append(meta["academic_year"])
            attribution = " | ".join(attr_parts)
            lines.append(f'> "{c["text"].strip()}"')
            lines.append(f'> -- {attribution}')
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cross-dimensional tools
    # ------------------------------------------------------------------

    @tool
    def browse_themes(
        tag_filters: Optional[dict] = None,
        theme_cluster: Optional[str] = None,
    ) -> str:
        """
        Browse the pre-extracted themes database. Returns themes, clusters, and
        key findings for documents matching the given filters.
        Use for open-ended exploration: "what themes appear in Gethen?" or
        "what are the main themes in Ekumen Outreach from 2024-25?".
        tag_filters: dict of tag key/value pairs, e.g. {"district": "Gethen"}.
        theme_cluster: optional theme cluster string.
        At least one filter is recommended to avoid returning the entire corpus.
        """
        if themes_df is None:
            return "Themes database not available -- run pipeline/deduplicate_themes.py first."

        df = themes_df[themes_df["theme_extraction_status"] == "ok"].copy()

        if tag_filters:
            def _matches(row):
                try:
                    tags = json.loads(row.get("tags") or "{}")
                except (json.JSONDecodeError, TypeError):
                    tags = {}
                return all(tags.get(k) == v for k, v in tag_filters.items() if v)
            df = df[df.apply(_matches, axis=1)]

        if theme_cluster:
            df = df[df["theme_clusters"].apply(
                lambda v: theme_cluster in _parse_json_list(v)
            )]

        if df.empty:
            return "No documents match those filters."

        shown = df.head(20)
        lines = [f"Found {len(df)} documents.\n"]

        for _, row in shown.iterrows():
            fname    = row.get("file_name", "")
            url      = row.get("drive_url", "")
            clusters = _parse_json_list(row.get("theme_clusters"))
            themes   = _parse_json_list(row.get("themes"))
            findings = _parse_json_list(row.get("key_findings"))
            title = f"[{fname}]({url})" if url else fname
            lines.append(f"**{title}**")
            if clusters: lines.append(f"  Clusters: {', '.join(clusters)}")
            if themes:   lines.append(f"  Themes:   {', '.join(themes)}")
            if findings: lines.append(f"  > {findings[0]}")
            lines.append("")

        if len(df) > 20:
            lines.append(f"... and {len(df) - 20} more documents.")

        return "\n".join(lines)

    _last_compare: dict[str, str] = {"passages": "", "topic": "", "dim": ""}

    @tool
    def compare(
        topic: str,
        dimension: str,
        tag_filters: Optional[dict] = None,
    ) -> str:
        """
        Compare how a topic appears across programs, districts, or academic years
        using semantic search across the document corpus.

        dimension: tag key to compare across -- "program", "district", or "academic_year"
        topic: the question or theme to search for

        tag_filters: optional dict to narrow the corpus before splitting by dimension.
        For example, {"program": "Ekumen Outreach"} to compare across districts,
        or {"doc_type": "site_visit"} to restrict to site visit documents.
        The dimension key is automatically excluded so all dimension values are retrieved.

        Returns retrieved passages grouped and cited by dimension value, ready for
        side-by-side synthesis.

        Use for:
          "How does teacher buy-in differ across programs?"
          "How has coaching support changed over time?"
          "How do Ansible Studies worlds differ in implementation?"
        """
        from collections import defaultdict

        dim = dimension.lower().strip()

        # Exclude the comparison dimension from filters so all its values are retrieved
        fixed_filters = {k: v for k, v in (tag_filters or {}).items() if k != dim} or None
        where = _build_where(fixed_filters, None)

        chunks = pipeline._retrieve(topic, where)
        if not chunks:
            return f"No relevant documents found for: {topic}"

        groups: dict[str, list] = defaultdict(list)
        for chunk in chunks:
            key = chunk["meta"].get(dim, "") or "Unknown"
            groups[key].append(chunk)

        if len(groups) < 2:
            single = list(groups.keys())[0] if groups else "unknown"
            return (
                f"All retrieved results are from a single {dim} ({single}). "
                f"Try broadening filters or using answer() for this {dim} directly."
            )

        # Chronological for academic_year; Unknown sorted last otherwise
        sorted_vals = sorted(groups.keys(), key=lambda v: (v == "Unknown", v or ""))

        noun = dim.replace("_", " ").title()
        lines = [f"**{noun} comparison -- {topic}**\n"]

        for val in sorted_vals:
            val_chunks = groups[val]
            lines.append(f"### {val}  ({len(val_chunks)} passages found)")
            for chunk in val_chunks[:4]:
                meta  = chunk["meta"]
                fname = meta.get("file_name", "")
                url   = url_lookup.get(fname, "") or meta.get("drive_url", "")
                link  = f"[{fname}]({url})" if url else fname
                attr_parts = [link]
                if dim != "district"      and meta.get("district"):      attr_parts.append(meta["district"])
                if dim != "academic_year" and meta.get("academic_year"): attr_parts.append(meta["academic_year"])
                lines.append(f'> "{chunk["text"].strip()}"')
                lines.append(f'> -- {" | ".join(attr_parts)}')
                lines.append("")
            lines.append("")

        result = "\n".join(lines)
        _last_compare["passages"] = result
        _last_compare["topic"]    = topic
        _last_compare["dim"]      = dim
        return result

    @tool
    def synthesize(
        topic: str,
        dimension: str,
    ) -> str:
        """
        Write a 2-4 sentence synthesis paragraph from the most recent compare() call.

        Identifies key similarities and differences across groups on the topic.
        Always call this immediately after compare() -- it reads the compare() result
        directly without requiring you to repeat the passages.

        topic: the same topic passed to compare()
        dimension: the same dimension passed to compare()

        Returns a synthesis paragraph suitable for leading the response.
        """
        passages = _last_compare.get("passages", "")
        if not passages:
            return "No compare() result available. Call compare() first, then synthesize()."

        noun = dimension.replace("_", " ")
        resp = _synth_llm.invoke([
            _SystemMessage(content=(
                "You write concise synthesis paragraphs from document research. "
                "Write 2-4 sentences identifying key similarities and differences "
                "across groups. Cite inline using ([filename](url)) links already "
                "present in the passages. Do not introduce new information. "
                "Do not use bullet points or headers."
            )),
            _HumanMessage(content=(
                f"Topic: {topic}\n"
                f"Comparing across: {noun}s\n\n"
                f"Retrieved passages:\n{passages}\n\n"
                f"Write a 2-4 sentence synthesis paragraph that leads with what "
                f"is consistent across {noun}s and then what differs. "
                f"Ground every claim in the passages above."
            )),
        ])
        return resp.content.strip()

    # ------------------------------------------------------------------
    # Survey stats tools
    # ------------------------------------------------------------------

    # Build index of spreadsheet files from metadata at construction time
    _survey_index: list[dict] = []
    if metadata:
        for rec in metadata:
            if rec.get("mime_type", "") in _SPREADSHEET_MIMES:
                try:
                    _tags = json.loads(rec.get("tags") or "{}")
                except (json.JSONDecodeError, TypeError):
                    _tags = {}
                _survey_index.append({
                    "file_name":  rec.get("file_name", ""),
                    "file_id":    rec.get("file_id", ""),
                    "mime_type":  rec.get("mime_type", ""),
                    "local_path": rec.get("local_path", ""),
                    "drive_url":  rec.get("drive_url", ""),
                    "tags":       _tags,
                })

    @tool
    def list_surveys(
        tag_filters: Optional[dict] = None,
    ) -> str:
        """
        List available survey spreadsheet files in the corpus.
        Use this before calling survey_stats() to discover which files exist
        and find the exact file name to query.
        tag_filters: optional dict of tag key/value pairs to filter,
        e.g. {"program": "Ekumen Outreach"}.
        """
        if not _survey_index:
            return "No survey spreadsheets found in the corpus."

        results = _survey_index
        if tag_filters:
            results = [
                r for r in results
                if all(r.get("tags", {}).get(k) == v for k, v in tag_filters.items() if v)
            ]

        if not results:
            return "No survey files match those filters."

        lines = [f"Found {len(results)} spreadsheet files:\n"]
        for r in results:
            parts = [f"**{r['file_name']}**"]
            for k, v in (r.get("tags") or {}).items():
                if v:
                    parts.append(f"{k}={v}")
            lines.append("- " + "  |  ".join(parts))
        return "\n".join(lines)

    @tool
    def survey_stats(
        file_name: str,
        question_fragment: Optional[str] = None,
        group_by: Optional[str] = None,
    ) -> str:
        """
        Return descriptive statistics for a survey spreadsheet.
        Use list_surveys() first to find the exact file name.

        For each question column returns:
          - Likert/categorical: response counts and percentages
          - Numeric (e.g. NPS 0-10): mean, median, min, max
          - Open-ended text columns are skipped (quantitative only)

        question_fragment (optional): filter to columns whose header contains
        this string, e.g. "facilitated" or "recommend". If omitted, all
        question columns are returned.

        group_by (optional): break stats out by a grouping column, e.g. "district",
        "role", or "school". Pass "?" to list available grouping columns first.
        """
        from pipeline.survey_stats import survey_stats_for_file

        # Find matching file -- normalize spaces/underscores and match bidirectionally
        def _normalize(s: str) -> str:
            return s.lower().replace("_", " ")

        needle = _normalize(file_name)
        matches = [
            r for r in _survey_index
            if needle in _normalize(r["file_name"])
            or _normalize(r["file_name"]) in needle
        ]
        if not matches:
            return f"No spreadsheet found matching '{file_name}'. Use list_surveys() to see available files."
        if len(matches) > 1:
            names = ", ".join(r["file_name"] for r in matches[:5])
            return f"Multiple files match '{file_name}': {names}. Please be more specific."

        rec  = matches[0]
        path = PROJECT_ROOT / rec["local_path"] if rec.get("local_path") else None

        if not path or not path.exists():
            file_id   = rec.get("file_id", "")
            mime_type = rec.get("mime_type", "")
            if not file_id:
                return f"File '{rec['file_name']}' is not available locally and has no Drive file ID."
            try:
                import tempfile as _tmp
                from pipeline.get_docs import build_drive_service, fetch_file_bytes
                service = build_drive_service()
                buf, ext, err = fetch_file_bytes(service, file_id, mime_type)
                if err:
                    return f"Could not download '{rec['file_name']}' from Google Drive: {err}"
                suffix = ext if ext.startswith(".") else f".{ext}"
                tmp_path = None
                try:
                    with _tmp.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(buf.read())
                        tmp_path = Path(tmp.name)
                    result = survey_stats_for_file(tmp_path, question_fragment, group_by, display_name=rec["file_name"])
                finally:
                    if tmp_path and tmp_path.exists():
                        tmp_path.unlink()
            except Exception as exc:
                return f"Error fetching '{rec['file_name']}' from Google Drive: {exc}"
        else:
            result = survey_stats_for_file(path, question_fragment, group_by, display_name=rec["file_name"])

        # Append the Drive link so the agent can cite the source correctly
        url = rec.get("drive_url", "")
        if url:
            result += f"\nSource: [{rec['file_name']}]({url})"

        return result

    return [
        search, answer, summarize, extract_quotes,
        browse_themes, compare, synthesize,
        list_surveys, survey_stats,
    ]


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _format_rag_result(result: dict, url_lookup: dict | None = None) -> str:
    answer = result["answer"]
    trailing: list[str] = []
    _lookup = url_lookup or {}

    for s in result.get("sources", []):
        url = _lookup.get(s["file_name"], "") or s.get("drive_url", "")
        link = f"[{s['file_name']}]({url})" if url else s["file_name"]
        tag = f"[{s['n']}]"
        if tag in answer:
            answer = answer.replace(tag, f"({link})")
        else:
            meta = " | ".join(f"{k}={v}" for k, v in s.get("tags", {}).items())
            trailing.append(f"  - {link}" + (f" -- {meta}" if meta else ""))

    if trailing:
        answer += "\n\nSources:\n" + "\n".join(trailing)

    return answer


def _parse_json_list(val) -> list[str]:
    if not val:
        return []
    try:
        parsed = json.loads(val) if isinstance(val, str) else val
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []