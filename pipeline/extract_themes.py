"""
Step 3.5: Theme Extractor

Reads documents.parquet, calls gpt-4o-mini once per document to extract:
  themes              3-5 short theme labels
  key_findings        2-3 key findings as complete sentences
  notable_quotes      1-3 exact verbatim quotes (character-for-character from source)
  inferred_academic_year  populated only when date_precision == "unknown"
  inferred_season         populated only when date_precision == "unknown"
  inferred_district       populated only when district is null in folder metadata

Writes themes_raw.parquet. Idempotent -- already-processed file_ids are skipped
so the script can be re-run safely after interruption.

Usage:
  python pipeline/extract_themes.py
"""

import json
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / "secrets" / ".env")

from providers import build_chat_model

DOCUMENTS_PATH = PROJECT_ROOT / "data" / "documents.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "themes_raw.parquet"

MAX_CHARS = 30_000   # truncate very long docs
MIN_CHARS = 100      # skip docs with too little text to extract themes from
REQUEST_DELAY = 0.1  # seconds between API calls

KNOWN_DISTRICTS = [
    "Gethen", "Anarres", "Urras", "Werel",
    "Hain", "Athshe", "Seggri", "Karhide",
    "Orgoreyn", "Davenant",
]

# Columns carried through from documents.parquet into themes_raw.parquet.
# Excludes `text` and `headings` (large, not needed downstream from themes).
CARRY_COLS = [
    "file_id", "file_name", "mime_type", "folder_path", "local_path", "drive_url",
    "program", "doc_type", "academic_year", "season", "date_precision", "district",
    "char_count", "extraction_status",
]

SYSTEM_PROMPT = """\
You analyze program documents from an organization's document library. Documents may include site visit notes, survey reflections, listening session summaries, evaluation reports, and other program data.

Your job is to extract structured information from each document. Return ONLY
a valid JSON object -- no markdown, no explanation, just JSON.\
"""


def _build_user_prompt(text: str, needs_date: bool, needs_district: bool) -> str:
    body = text[:MAX_CHARS]
    truncation_note = "\n\n[Document truncated]" if len(text) > MAX_CHARS else ""

    extra_keys = ""
    extra_fields = ""
    n = 4  # field counter for optional fields

    if needs_date:
        extra_fields += f"""
{n}. inferred_academic_year (string or null): The academic year this document
   appears to be from, formatted "YYYY-YY" (e.g. "2024-25"). Use July as the
   academic year boundary: July-December belong to the year starting that
   calendar year; January-June belong to the prior start year. Return null if
   the document gives no indication.

{n + 1}. inferred_season (string or null): "Fall" (July-December), "Spring"
   (January-June), or null if the document gives no indication.
"""
        extra_keys += ', "inferred_academic_year", "inferred_season"'
        n += 2

    if needs_district:
        districts = ", ".join(f'"{d}"' for d in KNOWN_DISTRICTS)
        extra_fields += f"""
{n}. inferred_district (string or null): The Hainish world this
   document appears to be about. Choose from: {districts}.
   Return null if the document spans multiple districts or gives no indication.
"""
        extra_keys += ', "inferred_district"'

    return f"""Extract the following from the document and return a JSON object with \
keys "themes", "key_findings", "notable_quotes"{extra_keys}.

1. themes (array of 2-6 word strings): 3-5 short labels capturing the main
   topics discussed. Use plain language, not jargon. Examples:
   "teacher buy-in", "data use in instruction", "implementation barriers".

2. key_findings (array of strings): 2-3 complete sentences describing the
   most important takeaways. Each sentence should stand alone.

3. notable_quotes (array of strings): 1-3 quotes that are EXACT, VERBATIM
   text copied character-for-character from the document. Do NOT paraphrase,
   summarize, shorten, or alter any wording. If no strong quote exists, return
   an empty array.
{extra_fields}
DOCUMENT:
{body}{truncation_note}"""


def _call_with_retry(llm: BaseChatModel, messages: list[dict], retries: int = 3) -> dict:
    lc_messages = [
        SystemMessage(content=m["content"]) if m["role"] == "system"
        else HumanMessage(content=m["content"])
        for m in messages
    ]
    for attempt in range(retries):
        try:
            response = llm.invoke(lc_messages)
            return json.loads(response.content)
        except Exception as e:
            if getattr(e, "code", None) == "insufficient_quota":
                raise
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def extract_themes(llm: BaseChatModel, record: dict) -> tuple[dict, str, str | None]:
    """
    Returns (fields_dict, status, error_message).
    status is "ok", "skipped", or "error".
    """
    text = record.get("text") or ""

    if record.get("extraction_status") != "ok" or len(text.strip()) < MIN_CHARS:
        return {}, "skipped", "Insufficient text"

    needs_date     = record.get("date_precision") == "unknown"
    needs_district = False

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_prompt(text, needs_date, needs_district)},
    ]

    try:
        result = _call_with_retry(llm, messages)
    except Exception:
        return {}, "error", traceback.format_exc(limit=3)

    fields = {
        "themes":         json.dumps(result.get("themes", []),         ensure_ascii=False),
        "key_findings":   json.dumps(result.get("key_findings", []),   ensure_ascii=False),
        "notable_quotes": json.dumps(result.get("notable_quotes", []), ensure_ascii=False),
        "inferred_academic_year": result.get("inferred_academic_year") if needs_date     else None,
        "inferred_season":        result.get("inferred_season")        if needs_date     else None,
        
    }
    return fields, "ok", None


def main() -> None:
    if not DOCUMENTS_PATH.exists():
        raise FileNotFoundError("documents.parquet not found -- run ingest.py first")

    df = pd.read_parquet(DOCUMENTS_PATH)

    # Resume: skip file_ids already successfully processed
    already_done: set[str] = set()
    if OUTPUT_PATH.exists():
        try:
            existing = pd.read_parquet(OUTPUT_PATH, columns=["file_id", "theme_extraction_status"])
            ok_rows = existing[existing["theme_extraction_status"] == "ok"]
            already_done = set(ok_rows["file_id"].tolist())
            if already_done:
                print(f"Resuming: {len(already_done)} already processed")
        except Exception:
            pass  # empty or schema-less file from a previous failed run -- start fresh

    to_process = df[~df["file_id"].isin(already_done)]
    print(f"Extracting themes from {len(to_process)} documents...")

    llm = build_chat_model()
    rows = []
    counts: dict[str, int] = {"ok": 0, "skipped": 0, "error": 0}

    for idx, (_, record) in enumerate(to_process.iterrows(), 1):
        rec = record.to_dict()
        fields, status, error = extract_themes(llm, rec)

        fname = rec.get("file_name", "")[:60]
        if status == "ok":
            try:
                theme_list = json.loads(fields.get("themes", "[]"))
                themes_str = ", ".join(theme_list) if theme_list else "(none)"
            except Exception:
                themes_str = fields.get("themes", "")
            print(f"  [{idx}/{len(to_process)}] {fname}")
            print(f"    themes: {themes_str}")
        elif status == "skipped":
            print(f"  [{idx}/{len(to_process)}] SKIP  {fname}")
        elif status == "error":
            last_line = error.strip().splitlines()[-1] if error and error.strip().splitlines() else "unknown"
            print(f"  [{idx}/{len(to_process)}] ERROR  {fname}")
            print(f"    {last_line}")
            if "insufficient_quota" in (error or ""):
                print("\nQuota exhausted -- add credits at https://platform.openai.com/settings/billing")
                print("Re-run this script after adding credits; already-processed docs will be skipped.")
                break

        counts[status] = counts.get(status, 0) + 1

        row = {col: rec.get(col) for col in CARRY_COLS}
        row.update({
            "themes":                  fields.get("themes", "[]"),
            "key_findings":            fields.get("key_findings", "[]"),
            "notable_quotes":          fields.get("notable_quotes", "[]"),
            "inferred_academic_year":  fields.get("inferred_academic_year"),
            "inferred_season":         fields.get("inferred_season"),
            "inferred_district":       fields.get("inferred_district"),
            "theme_extraction_status": status,
            "theme_extraction_error":  error,
        })
        rows.append(row)

        if status == "ok":
            time.sleep(REQUEST_DELAY)

    # Merge new rows with previously successful rows (drop old errors -- they were retried)
    new_df = pd.DataFrame(rows)
    if already_done and OUTPUT_PATH.exists():
        try:
            existing_df = pd.read_parquet(OUTPUT_PATH)
            if "theme_extraction_status" in existing_df.columns:
                existing_df = existing_df[existing_df["theme_extraction_status"] == "ok"]
            new_df = pd.concat([existing_df, new_df], ignore_index=True)
        except Exception:
            pass

    new_df.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nDone.")
    print(f"  OK:       {counts.get('ok', 0)}")
    print(f"  Skipped:  {counts.get('skipped', 0)}")
    print(f"  Errors:   {counts.get('error', 0)}")
    print(f"  Output:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
