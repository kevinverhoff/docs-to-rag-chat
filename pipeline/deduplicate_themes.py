"""
Step 3.6: Theme Deduplicator

Builds a three-level theme hierarchy and writes:
  - theme_map.json       raw label -> canonical label  (human-editable)
  - theme_clusters.json  canonical label -> cluster    (human-editable)
  - themes.parquet       final themes layer with all three levels

themes.parquet theme columns:
  themes_raw      JSON array -- original labels as extracted (never altered)
  themes          JSON array -- canonical labels after dedup
  theme_clusters  JSON array -- unique superordinate clusters for this document

ONLY the themes hierarchy is derived from LLM calls. key_findings and
notable_quotes are copied through completely unchanged from themes_raw.parquet.

If a mapping file already exists its LLM call is skipped -- edit the file
and re-run to apply changes without regenerating from scratch.

Usage:
  python pipeline/deduplicate_themes.py
"""

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

PROJECT_ROOT       = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")
THEMES_RAW_PATH    = PROJECT_ROOT / "data" / "themes_raw.parquet"
THEME_MAP_PATH     = PROJECT_ROOT / "data" / "theme_map.json"
THEME_CLUSTERS_PATH = PROJECT_ROOT / "data" / "theme_clusters.json"
OUTPUT_PATH        = PROJECT_ROOT / "data" / "themes.parquet"

MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_json_list(val) -> list[str]:
    if not val or val == "[]":
        return []
    try:
        result = json.loads(val)
        return [t for t in result if isinstance(t, str) and t.strip()]
    except (json.JSONDecodeError, TypeError):
        return []


def _collect_unique(df: pd.DataFrame, col: str) -> list[str]:
    unique: set[str] = set()
    for val in df[col]:
        unique.update(t.strip() for t in _parse_json_list(val))
    return sorted(unique)

# ---------------------------------------------------------------------------
# Pass 1: raw -> canonical (theme_map.json)
# ---------------------------------------------------------------------------

_MAP_SYSTEM = """\
You are a data analyst normalizing theme labels from an education research
corpus. Group synonymous or near-synonymous labels into clean canonical labels.

Rules:
- Canonical labels must be 2-5 words, plain language, no jargon
- Preserve meaningful distinctions -- do not over-merge unrelated concepts
- Every input label must appear as a key in the output
- A label with no close synonym should map to a lightly cleaned version of itself
- Return ONLY a valid JSON object: { "raw label": "canonical label", ... }\
"""


def _map_prompt(themes: list[str]) -> str:
    return (
        f"Below are {len(themes)} theme labels extracted from "
        "program documents (site visits, surveys, listening sessions, evaluation "
        "reports, district data). Group synonyms into canonical labels.\n\n"
        "THEME LABELS:\n" + "\n".join(f"- {t}" for t in themes)
    )


def generate_theme_map(client: OpenAI, themes: list[str]) -> dict[str, str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _MAP_SYSTEM},
            {"role": "user",   "content": _map_prompt(themes)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    mapping: dict[str, str] = json.loads(resp.choices[0].message.content)
    return {t: mapping.get(t, t) for t in themes}

# ---------------------------------------------------------------------------
# Pass 2: canonical -> cluster (theme_clusters.json)
# ---------------------------------------------------------------------------

_CLUSTER_SYSTEM = """\
You are a data analyst building a thematic taxonomy for an education research
corpus. Assign each canonical theme label to a broad superordinate cluster.

Rules:
- Produce 5-8 clusters that span the full set of themes
- Cluster names should be 2-5 words, descriptive, and mutually exclusive
- Every canonical label must appear as a key in the output
- Return ONLY a valid JSON object: { "canonical label": "Cluster Name", ... }\
"""


def _cluster_prompt(canonical: list[str]) -> str:
    return (
        f"Assign each of these {len(canonical)} canonical theme labels from "
        "program documents to a broad superordinate cluster. "
        "Produce 5-8 clusters total.\n\n"
        "CANONICAL LABELS:\n" + "\n".join(f"- {t}" for t in canonical)
    )


def generate_cluster_map(client: OpenAI, canonical: list[str]) -> dict[str, str]:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _CLUSTER_SYSTEM},
            {"role": "user",   "content": _cluster_prompt(canonical)},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    mapping: dict[str, str] = json.loads(resp.choices[0].message.content)
    return {t: mapping.get(t, t) for t in canonical}

# ---------------------------------------------------------------------------
# Apply mappings
# ---------------------------------------------------------------------------

def _apply_theme_map(raw_json: str, theme_map: dict[str, str]) -> str:
    """Map raw labels to canonical, deduplicate within the list."""
    raw = _parse_json_list(raw_json)
    seen: set[str] = set()
    canonical: list[str] = []
    for t in raw:
        mapped = theme_map.get(t, t)
        if mapped not in seen:
            seen.add(mapped)
            canonical.append(mapped)
    return json.dumps(canonical, ensure_ascii=False)


def _apply_cluster_map(canonical_json: str, cluster_map: dict[str, str]) -> str:
    """Map canonical labels to clusters, return unique clusters for this document."""
    canonical = _parse_json_list(canonical_json)
    seen: set[str] = set()
    clusters: list[str] = []
    for t in canonical:
        cluster = cluster_map.get(t, "Uncategorized")
        if cluster not in seen:
            seen.add(cluster)
            clusters.append(cluster)
    return json.dumps(clusters, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not THEMES_RAW_PATH.exists():
        raise FileNotFoundError(
            "themes_raw.parquet not found -- run extract_themes.py first"
        )

    df = pd.read_parquet(THEMES_RAW_PATH)
    ok = df[df["theme_extraction_status"] == "ok"]
    print(f"Loaded {len(df)} rows ({len(ok)} with extracted themes)")

    client = OpenAI()

    # -- Pass 1: raw -> canonical -------------------------------------------

    if THEME_MAP_PATH.exists():
        print(f"theme_map.json found -- skipping LLM call (delete to regenerate)")
        with open(THEME_MAP_PATH, encoding="utf-8") as f:
            theme_map: dict[str, str] = json.load(f)
        print(f"  {len(theme_map)} entries")
    else:
        unique_raw = _collect_unique(ok, "themes")
        print(f"Found {len(unique_raw)} unique raw theme labels")
        print("Pass 1: generating canonical mapping...")
        theme_map = generate_theme_map(client, unique_raw)
        with open(THEME_MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(theme_map, f, indent=2, ensure_ascii=False)
        print(f"  Saved {THEME_MAP_PATH.name}")

    # Apply raw -> canonical to get the canonical label set
    canonical_series = df["themes"].apply(lambda v: _apply_theme_map(v, theme_map))
    unique_canonical = sorted({
        t for val in canonical_series for t in _parse_json_list(val)
    })
    print(f"  {len(unique_canonical)} unique canonical labels")

    # -- Pass 2: canonical -> cluster ---------------------------------------

    if THEME_CLUSTERS_PATH.exists():
        print(f"theme_clusters.json found -- skipping LLM call (delete to regenerate)")
        with open(THEME_CLUSTERS_PATH, encoding="utf-8") as f:
            cluster_map: dict[str, str] = json.load(f)
        print(f"  {len(cluster_map)} entries")
    else:
        print("Pass 2: generating cluster mapping...")
        cluster_map = generate_cluster_map(client, unique_canonical)
        with open(THEME_CLUSTERS_PATH, "w", encoding="utf-8") as f:
            json.dump(cluster_map, f, indent=2, ensure_ascii=False)
        print(f"  Saved {THEME_CLUSTERS_PATH.name}")
        clusters_used = sorted(set(cluster_map.values()))
        print(f"  {len(clusters_used)} clusters: {', '.join(clusters_used)}")

    # -- Build themes.parquet -----------------------------------------------

    out = df.copy()

    # Preserve raw themes before overwriting
    out["themes_raw"] = out["themes"]

    # Apply mappings to themes only -- key_findings and notable_quotes untouched
    out["themes"]         = canonical_series
    out["theme_clusters"] = out["themes"].apply(
        lambda v: _apply_cluster_map(v, cluster_map)
    )

    # Promote LLM-inferred dates where folder path gave none
    date_mask = (
        (out["date_precision"] == "unknown")
        & out["inferred_academic_year"].notna()
    )
    if date_mask.any():
        out.loc[date_mask, "academic_year"]  = out.loc[date_mask, "inferred_academic_year"]
        out.loc[date_mask, "season"]         = out.loc[date_mask, "inferred_season"]
        out.loc[date_mask, "date_precision"] = "llm_inferred"
        print(f"  Promoted inferred dates for {date_mask.sum()} documents")

    # Promote LLM-inferred district where folder path gave none
    if "inferred_district" in out.columns:
        dist_mask = out["district"].isna() & out["inferred_district"].notna()
        if dist_mask.any():
            out.loc[dist_mask, "district"] = out.loc[dist_mask, "inferred_district"]
            print(f"  Promoted inferred district for {dist_mask.sum()} documents")

    out.to_parquet(OUTPUT_PATH, index=False)

    # -- Summary ------------------------------------------------------------

    cluster_counts: dict[str, int] = {}
    for val in out.loc[out["theme_extraction_status"] == "ok", "theme_clusters"]:
        for c in _parse_json_list(val):
            cluster_counts[c] = cluster_counts.get(c, 0) + 1

    print(f"\nDone. Output: {OUTPUT_PATH}")
    print(f"\nDocuments per cluster:")
    for cluster, count in sorted(cluster_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>3}  {cluster}")


if __name__ == "__main__":
    main()