"""
Step 9: Streamlit chat interface for the document library.

Run with:
  python -m streamlit run app.py
"""

import json
import re
from collections import Counter
from pathlib import Path
import base64

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from feedback import feedback_form

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")

THEMES_PATH = PROJECT_ROOT / "data" / "themes.parquet"

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------

_FAV_PNG = PROJECT_ROOT / "media" / "Favicon.png"
_LOGO_PNG = PROJECT_ROOT / "media" / "logo.png"

# Use the separate favicon image for the browser tab.
page_icon = str(_FAV_PNG) if _FAV_PNG.exists() else "🤖"

st.set_page_config(
    page_title="Your Personal Research Assistant",
    page_icon=page_icon,
    layout="wide",
)


def _get_logo_b64() -> str | None:
    """Return base64-encoded bytes for the sidebar logo (use logo.png if present)."""
    if _LOGO_PNG.exists():
        try:
            return base64.b64encode(_LOGO_PNG.read_bytes()).decode()
        except Exception:
            pass
    return None


# Prepare a base64 favicon/logo to embed in the page head and sidebar
_LOGO_B64 = _get_logo_b64()
_fav_dir = PROJECT_ROOT / "media" / "favicons"
if _fav_dir.exists():
    links = []
    mapping = [
        ("favicon-512.png", "512x512"),
        ("favicon-256.png", "256x256"),
        ("favicon-192.png", "192x192"),
        ("apple-touch-icon-180.png", "180x180"),
        ("favicon-48.png", "48x48"),
        ("favicon-32.png", "32x32"),
        ("favicon-16.png", "16x16"),
    ]
    for fname, size in mapping:
        p = _fav_dir / fname
        if p.exists():
            try:
                b64 = base64.b64encode(p.read_bytes()).decode()
                links.append(f'<link rel="icon" href="data:image/png;base64,{b64}" sizes="{size}" type="image/png">')
            except Exception:
                pass
    if links:
        st.markdown("\n".join(links), unsafe_allow_html=True)
elif _LOGO_B64:
    # Inject a data-uri favicon (multiple sizes) so browsers use the provided image
    st.markdown(
        f"""
        <link rel="icon" href="data:image/png;base64,{_LOGO_B64}" sizes="32x32" type="image/png">
        <link rel="icon" href="data:image/png;base64,{_LOGO_B64}" sizes="48x48" type="image/png">
        <link rel="apple-touch-icon" href="data:image/png;base64,{_LOGO_B64}">
        """,
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Cached resources
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge base...")
def get_agent():
    from rag_pipeline import RagPipeline
    from agent import Agent
    return Agent(RagPipeline())


@st.cache_data(show_spinner=False)
def get_themes() -> pd.DataFrame | None:
    if not THEMES_PATH.exists():
        return None
    df = pd.read_parquet(THEMES_PATH)
    if "theme_extraction_status" in df.columns:
        df = df[df["theme_extraction_status"] == "ok"]
    return df if not df.empty else None


@st.cache_data(show_spinner=False)
def get_url_lookup() -> dict[str, str]:
    """file_name -> drive_url, built from themes.parquet so we never rely on the LLM to relay URLs."""
    if not THEMES_PATH.exists():
        return {}
    try:
        df = pd.read_parquet(THEMES_PATH, columns=["file_name", "drive_url"])
        return {row["file_name"]: row["drive_url"] for _, row in df.iterrows() if row.get("drive_url")}
    except Exception:
        return {}


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def _distinct(df: pd.DataFrame | None, col: str) -> list[str]:
    if df is None or col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())


def _distinct_clusters(df: pd.DataFrame | None) -> list[str]:
    if df is None or "theme_clusters" not in df.columns:
        return []
    seen: set[str] = set()
    for val in df["theme_clusters"].dropna():
        try:
            items = json.loads(val) if isinstance(val, str) else val
            if isinstance(items, list):
                seen.update(str(x) for x in items)
        except Exception:
            pass
    return sorted(seen)


def sidebar(themes_df: pd.DataFrame | None) -> dict:
    with st.sidebar:
        # Display the local sidebar logo from the media folder.
        if _LOGO_B64:
            st.markdown(
                f'<div style="text-align:center;margin:-32px 0 0 0;padding:0;">'
                f'<img src="data:image/png;base64,{_LOGO_B64}" '
                f'style="width:100%;max-width:380px;height:auto;display:block;margin:0 auto;"/>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='text-align:center;padding:16px 0 8px 0;'>"
                "<div style='font-size:3.5rem;line-height:1;'>&#x1F916;</div>"
                "<div style='margin-top:10px;color:#24265b;font-size:1.1rem;"
                "font-weight:700;letter-spacing:0.02em;'>Your Personal Research Assistant</div>"
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            "<div style='margin:0;padding:0;'>"
            "<hr style='margin:0 0 16px 0;border:none;border-bottom:2px solid #f04c24;'/>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<div style='margin:0;padding:0;color:#0e3350;font-family: \"Bebas Neue\", sans-serif; font-weight:700;font-size:1rem;letter-spacing:0.03em;'>FILTERS</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='margin:10px 0 18px 0;color:#5C628F;font-size:0.88rem;line-height:1.55;'>Scope the search to a specific subset of documents.</p>",
            unsafe_allow_html=True,
        )

        def _fmt_doc_type(v: str) -> str:
            return v if v == "(All)" else v.replace("_", " ").title()

        st.markdown(
            "<div class='sidebar-field-label' style='margin:0 0 8px 0;'>Program</div>",
            unsafe_allow_html=True,
        )
        program = st.selectbox("", ["(All)"] + _distinct(themes_df, "program"), key="program", label_visibility="collapsed")

        st.markdown(
            "<div class='sidebar-field-label' style='margin:14px 0 8px 0;'>District</div>",
            unsafe_allow_html=True,
        )
        district = st.selectbox("", ["(All)"] + _distinct(themes_df, "district"), key="district", label_visibility="collapsed")

        st.markdown(
            "<div class='sidebar-field-label' style='margin:14px 0 8px 0;'>Academic Year</div>",
            unsafe_allow_html=True,
        )
        academic_year = st.selectbox("", ["(All)"] + _distinct(themes_df, "academic_year"), key="academic_year", label_visibility="collapsed")

        st.markdown(
            "<div class='sidebar-field-label' style='margin:14px 0 8px 0;'>Doc Type</div>",
            unsafe_allow_html=True,
        )
        doc_type = st.selectbox("", ["(All)"] + _distinct(themes_df, "doc_type"), format_func=_fmt_doc_type, key="doc_type", label_visibility="collapsed")

        st.markdown(
            "<div class='sidebar-field-label' style='margin:14px 0 8px 0;'>Theme Cluster</div>",
            unsafe_allow_html=True,
        )
        theme_cluster = st.selectbox("", ["(All)"] + _distinct_clusters(themes_df), key="theme_cluster", label_visibility="collapsed")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.pop("messages", None)
            st.rerun()

    return {
        "program":       None if program       == "(All)" else program,
        "district":      None if district      == "(All)" else district,
        "academic_year": None if academic_year == "(All)" else academic_year,
        "doc_type":      None if doc_type      == "(All)" else doc_type,
        "theme_cluster": None if theme_cluster == "(All)" else theme_cluster,
    }


# ------------------------------------------------------------------
# Source parsing + rendering
# ------------------------------------------------------------------

def _split_answer_sources(text: str) -> tuple[str, list[dict]]:
    """
    Citations are now inline markdown links inside the answer text itself.
    This function is kept for API compatibility but simply passes the text through.
    """
    return text, []


def _render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    url_lookup = get_url_lookup()
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for s in sources:
            url = s["drive_url"] or url_lookup.get(s["file_name"], "")
            if url:
                st.markdown(f"**[{s['n']}]** [{s['file_name']}]({url})")
            else:
                st.markdown(f"**[{s['n']}]** {s['file_name']}")


# ------------------------------------------------------------------
# Debug trace
# ------------------------------------------------------------------

def _extract_trace(messages: list) -> tuple:
    """Extract tool call + result pairs and token usage from agent message list.

    Returns (trace, usage) where usage = {"input_tokens": int, "output_tokens": int}
    summed across all AIMessages in the turn.
    """
    tool_results: dict[str, str] = {}
    for msg in messages:
        call_id = getattr(msg, "tool_call_id", None)
        if call_id:
            tool_results[call_id] = getattr(msg, "content", "")

    trace = []
    usage = {"input_tokens": 0, "output_tokens": 0}
    for msg in messages:
        meta = getattr(msg, "usage_metadata", None)
        if isinstance(meta, dict):
            usage["input_tokens"]  += meta.get("input_tokens", 0)
            usage["output_tokens"] += meta.get("output_tokens", 0)
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                trace.append({
                    "tool":   tc.get("name", ""),
                    "args":   tc.get("args", {}),
                    "result": tool_results.get(tc.get("id", ""), ""),
                })
    return trace, usage


def _render_debug(trace: list[dict], usage: dict = None) -> None:
    if not trace:
        return
    label = f"\U0001f50d Agent trace — {len(trace)} tool call(s)"
    if usage and (usage.get("input_tokens") or usage.get("output_tokens")):
        label += f"  ·  {usage['input_tokens']:,} in / {usage['output_tokens']:,} out tokens"
    with st.expander(label, expanded=False):
        for i, step in enumerate(trace, 1):
            st.markdown(f"**Step {i} · `{step['tool']}`**")
            if step["args"]:
                st.markdown("*Arguments*")
                st.json(step["args"])
            result_text = step["result"]
            if result_text:
                truncated = result_text if len(result_text) <= 1500 else result_text[:1500] + "\n…[truncated]"
                st.markdown("*Tool output*")
                st.code(truncated, language="text")
            if i < len(trace):
                st.divider()


# ------------------------------------------------------------------
# Chat tab
# ------------------------------------------------------------------

def _filter_label(filters: dict) -> str:
    parts = [
        f"{k.replace('_', ' ').title()}: **{v}**"
        for k, v in filters.items() if v
    ]
    return " · ".join(parts) if parts else "No filters active — searching all documents."


def chat_tab(ag, filters: dict) -> None:
    if any(filters.values()):
        st.caption(_filter_label(filters))

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Layout order determines visual position:
    #   msg_container  → messages (grows upward as history accumulates)
    #   warning_slot   → filter-change banner (sits just above the chat input)
    #   chat_input     → sticky at bottom of viewport
    msg_container = st.container()
    warning_slot  = st.container()
    prompt = st.chat_input("Ask a question...")

    # Fill warning slot — renders just above the chat input, visible without scrolling
    _active = st.session_state.get("active_filters")
    if _active and _active != filters and st.session_state.messages:
        changes = []
        for _k in ("program", "district", "academic_year", "doc_type", "theme_cluster"):
            _old, _new = _active.get(_k), filters.get(_k)
            if _old != _new:
                changes.append(
                    f"{_k.replace('_', ' ').title()}: "
                    f"**{_old or 'All'}** → **{_new or 'All'}**"
                )
        with warning_slot.container(border=True):
            _c1, _c2 = st.columns([5, 1])
            with _c1:
                st.warning(
                    "**Filters changed mid-conversation.** "
                    "Previous responses used a different scope — the history may no longer "
                    "match your current question. Consider starting a fresh conversation.\n\n"
                    + "  ·  ".join(changes),
                    icon="⚠️",
                )
            with _c2:
                st.markdown("&nbsp;", unsafe_allow_html=True)
                if st.button("Clear chat", key="_filter_clear", use_container_width=True):
                    st.session_state.pop("messages", None)
                    st.session_state.pop("active_filters", None)
                    st.rerun()

    with msg_container:
        for i, msg in enumerate(st.session_state.messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    _render_sources(msg["sources"])
                if msg.get("trace"):
                    _render_debug(msg["trace"], msg.get("usage"))
            if msg["role"] == "assistant":
                question = next(
                    (m["content"] for m in reversed(st.session_state.messages[:i]) if m["role"] == "user"),
                    "",
                )
                feedback_form(i, question, msg["content"])

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.active_filters = filters
        with msg_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    history = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages[:-1]
                    ]
                    result = ag.chat(prompt, history=history, **filters)

                answer, sources = _split_answer_sources(result["answer"])
                trace, usage = _extract_trace(result.get("messages", []))
                st.markdown(answer)
                _render_sources(sources)
                _render_debug(trace, usage)

        new_index = len(st.session_state.messages)
        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
            "trace":   trace,
            "usage":   usage,
        })
        with msg_container:
            feedback_form(new_index, prompt, answer)


# ------------------------------------------------------------------
# Browse tab
# ------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    for col, val in [
        ("program",       filters.get("program")),
        ("district",      filters.get("district")),
        ("academic_year", filters.get("academic_year")),
        ("doc_type",      filters.get("doc_type")),
    ]:
        if val and col in df.columns:
            df = df[df[col] == val]

    cluster = filters.get("theme_cluster")
    if cluster and "theme_clusters" in df.columns:
        def _has(val):
            try:
                items = json.loads(val) if isinstance(val, str) else val
                return cluster in (items or [])
            except Exception:
                return False
        df = df[df["theme_clusters"].apply(_has)]

    return df


def _parse_list(val) -> list[str]:
    if not val:
        return []
    try:
        items = json.loads(val) if isinstance(val, str) else val
        return [str(x) for x in items] if isinstance(items, list) else []
    except Exception:
        return []


def browse_tab(themes_df: pd.DataFrame | None, filters: dict) -> None:
    if themes_df is None:
        st.warning(
            "Themes database not found. "
            "Run `python pipeline/__init__.py` through Step 3.6 to build it."
        )
        return

    df = _apply_filters(themes_df.copy(), filters)
    if df.empty:
        st.info("No documents match the current filters.")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        group_col = st.selectbox(
            "Group by",
            ["program", "academic_year", "district", "doc_type"],
            key="browse_group",
        )
    with col2:
        st.metric("Documents", len(df))

    if group_col not in df.columns:
        st.warning(f"'{group_col}' column not available.")
        return

    groups = sorted(df.groupby(group_col, dropna=True), key=lambda x: str(x[0]))

    for name, group in groups:
        if not name:
            continue

        all_themes: list[str] = []
        for _, row in group.iterrows():
            all_themes.extend(_parse_list(row.get("themes")))
        top_themes = [t for t, _ in Counter(all_themes).most_common(6)]

        with st.expander(f"**{name}** — {len(group)} documents"):
            if top_themes:
                st.markdown("**Top themes:** " + "  ·  ".join(top_themes))
                st.divider()

            for _, row in group.iterrows():
                fname    = row.get("file_name", "")
                url      = row.get("drive_url", "")
                clusters = _parse_list(row.get("theme_clusters"))
                themes   = _parse_list(row.get("themes"))
                findings = _parse_list(row.get("key_findings"))

                header = f"[{fname}]({url})" if url else fname
                st.markdown(f"**{header}**")
                if clusters:
                    st.caption("Clusters: " + " · ".join(clusters))
                if themes:
                    st.markdown("Themes: " + ", ".join(themes))
                if findings:
                    st.markdown(f"> {findings[0]}")
                st.markdown("---")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _inject_fonts() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Nunito:ital,wght@0,200..1000;1,200..1000&family=Teachers:ital,wght@0,400..900;1,400..900&display=swap');

        /* Body — Nunito */
        html, body, [class*="stMarkdown"], p, li, td, th, label,
        .stChatMessage, .stTextInput, .stSelectbox, .stCaption,
        div[data-testid="stChatMessageContent"] {
            font-family: 'Nunito', sans-serif !important;
        }

        /* Primary headings — Teachers */
        h1, h2,
        [class*="stMarkdown"] h1,
        [class*="stMarkdown"] h2 {
            font-family: 'Teachers', sans-serif !important;
        }

        /* Sub-headings / condensed — Bebas Neue */
        h3, h4, h5, h6,
        [class*="stMarkdown"] h3,
        [class*="stMarkdown"] h4,
        [class*="stMarkdown"] h5,
        [class*="stMarkdown"] h6 {
            font-family: 'Bebas Neue', sans-serif !important;
            letter-spacing: 0.04em;
        }

        /* Soft sidebar background */
        section[data-testid="stSidebar"] > div {
            background-color: #F8F5EF !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #F8F5EF !important;
        }

        /* Sidebar selectbox / dropdown polish */
        section[data-testid="stSidebar"] .stSelectbox > div > div > div > div {
            border-radius: 14px !important;
        }
        section[data-testid="stSidebar"] .stSelectbox > div > div > select {
            min-height: 3.1rem !important;
            border-radius: 14px !important;
            color: #18334c !important;
        }
        section[data-testid="stSidebar"] .stSelectbox {
            margin-bottom: 18px !important;
        }

        /* Search box appearance */
        .stTextInput>div>div>input,
        div[role="textbox"],
        .stTextArea>div>div>textarea {
            min-height: 3.4rem !important;
            border-radius: 16px !important;
            padding: 0.8rem 1rem !important;
            box-shadow: 0 12px 24px rgba(0, 0, 0, 0.08) !important;
            border: 1px solid rgba(16, 40, 62, 0.12) !important;
        }
        .stTextInput>div>div>input::placeholder,
        .stTextArea>div>div>textarea::placeholder,
        div[role="textbox"]::placeholder {
            color: #6e7a8d !important;
            opacity: 1 !important;
        }

        /* Active tab underline */
        button[role="tab"][aria-selected="true"] {
            border-bottom: 3px solid #f04c24 !important;
            padding-bottom: 10px !important;
            margin: 0 10px !important;
        }

        /* Strengthen filter label hierarchy */
        section[data-testid="stSidebar"] .sidebar-field-label {
            font-weight: 500 !important;
            color: #24265b !important;
        }

        div[role="tablist"] {
            margin-top: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_fonts()
    themes_df = get_themes()
    filters   = sidebar(themes_df)

    ag = None
    try:
        ag = get_agent()
    except Exception as exc:
        st.error(
            f"Could not load the knowledge base: {exc}\n\n"
            "Run `python pipeline/__init__.py` to build the vector store and themes first."
        )

    st.markdown(
        """
        <div style='margin-bottom:16px;'>
          <div style='margin:0;color:#24265b;font-size:2rem;font-weight:700;letter-spacing:-0.02em;'>🤖 Your Personal Research Assistant</div>
          <div style='margin:4px 0 0 0;color:#5C628F;font-size:1rem;line-height:1.4;'>Search across the document library.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_t, browse_t = st.tabs(["💬 Chat", "📊 Browse Themes"])

    with chat_t:
        if ag is not None:
            chat_tab(ag, filters)
        else:
            st.info("Knowledge base not available. See the error above.")

    with browse_t:
        browse_tab(themes_df, filters)


if __name__ == "__main__":
    main()
