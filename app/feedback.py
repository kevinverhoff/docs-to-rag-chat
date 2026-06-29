"""
Feedback collection and Google Sheets writing for the document research assistant.

Rubric dimensions (each scored 1-4):
  Answer Relevance       15%
  Evidence Grounding     25%
  Citation Quality       20%
  Synthesis Quality      15%
  Theme Validity         10%
  5 Conditions Alignment 10%
  Usefulness              5%

The weighted composite score is written alongside raw dimension scores to the
configured Google Sheet tab.
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import gspread
import streamlit as st
from google.oauth2.service_account import Credentials

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
_CREDS_PATH  = PROJECT_ROOT / "secrets" / "google-service-account.json"
_SHEET_ID    = ""  # Replace with your Google Sheet ID
_TAB_GID     = 0  # Replace with your target worksheet GID

_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

# Dimension names must match the sheet column headers (minus the " (1-4)" suffix)
DIMENSIONS: list[tuple[str, float]] = [
    ("Answer Relevance",       0.15),
    ("Evidence Grounding",     0.25),
    ("Citation Quality",       0.20),
    ("Synthesis Quality",      0.15),
    ("Theme Validity",         0.10),
    ("Framework Alignment",      0.10),
    ("Usefulness",             0.05),
]

# Sheet columns: Question ID | Question | <dim> (1-4) x7 | Weighted Score | Evaluator Notes
# Row 1 = headers, Row 2 = descriptions — data starts at row 3

SCORE_LABELS = {
    1: "1 — Poor",
    2: "2 — Fair",
    3: "3 — Good",
    4: "4 — Excellent",
}

# ---------------------------------------------------------------------------
# Sheets client (cached per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _get_worksheet():
    creds = Credentials.from_service_account_file(str(_CREDS_PATH), scopes=_SCOPES)
    gc    = gspread.authorize(creds)
    sh    = gc.open_by_key(_SHEET_ID)
    # Resolve by gid so renaming the tab doesn't break the integration
    for ws in sh.worksheets():
        if ws.id == _TAB_GID:
            return ws
    # Fallback: first sheet
    return sh.sheet1


def _next_question_id(ws) -> str:
    """Return the next Q-### ID based on existing rows (data starts at row 3)."""
    all_ids = [
        r[0] for r in ws.get_all_values()[2:]  # skip header + description rows
        if r and r[0].startswith("Q-")
    ]
    if not all_ids:
        return "Q-001"
    last = max(int(qid.split("-")[1]) for qid in all_ids if qid.split("-")[1].isdigit())
    return f"Q-{last + 1:03d}"


def write_feedback(
    question: str,
    answer: str,
    scores: dict[str, int],
    notes: str,
) -> None:
    """Append one feedback row to the Google Sheet."""
    ws = _get_worksheet()

    weighted = round(sum(scores.get(dim, 0) * weight for dim, weight in DIMENSIONS), 2)
    qid      = _next_question_id(ws)

    row = (
        [qid, question]
        + [scores.get(dim, "") for dim, _ in DIMENSIONS]
        + [weighted, notes, answer]
    )
    ws.append_row(row, value_input_option="USER_ENTERED")


# ---------------------------------------------------------------------------
# Streamlit UI component
# ---------------------------------------------------------------------------

def feedback_form(message_index: int, question: str, answer: str) -> None:
    """
    Render a 'Rate response' button for a single assistant message.
    When clicked, an inline scoring form expands.

    Call once per assistant message, passing a unique message_index.
    """
    thumb_key  = f"_fb_thumb_{message_index}"
    submit_key = f"_fb_submitted_{message_index}"

    if st.session_state.get(submit_key):
        st.caption("✓ Feedback submitted")
        return

    col_btn, col_spacer = st.columns([2, 10])
    with col_btn:
        if st.button("Provide feedback", key=f"_fb_up_{message_index}"):
            st.session_state[thumb_key] = True

    if not st.session_state.get(thumb_key):
        return

    with st.expander("Rate this response", expanded=True):
        scores: dict[str, int] = {}
        for dim, weight in DIMENSIONS:
            pct  = int(weight * 100)
            val  = st.select_slider(
                f"{dim} ({pct}%)",
                options=[1, 2, 3, 4],
                format_func=lambda x: SCORE_LABELS[x],
                value=3,
                key=f"_fb_{message_index}_{dim}",
            )
            scores[dim] = val

        notes = st.text_area(
            "Notes (optional)",
            placeholder="Any specific observations about this response...",
            key=f"_fb_notes_{message_index}",
        )

        if st.button("Submit feedback", key=f"_fb_submit_{message_index}", type="primary"):
            try:
                write_feedback(question, answer, scores, notes)
                st.session_state[submit_key] = True
                st.rerun()
            except Exception as exc:
                st.error(f"Could not write to Google Sheet: {exc}")
