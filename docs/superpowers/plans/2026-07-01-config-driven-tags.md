# Config-Driven Tags Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all hardcoded org-specific values (program names, district lists, sidebar labels) from Python source files and move them into a single `tags_config.yaml` that any forker can edit without touching code.

**Architecture:** A new `tags_config.yaml` at the project root defines folder-to-tag mappings for the GDrive source and which tag keys appear as app filters. `config.py` loads this file and exposes the values. The pipeline (ingest to vectorstore) stores tags as a JSON column instead of individual named columns, so the schema is not hard-coupled to any particular taxonomy. The app sidebar and filter interface are driven by config keys rather than hardcoded field names.

**Tech Stack:** Python 3.11+, PyYAML (new dep), pandas, ChromaDB, Streamlit, LangChain

## Global Constraints

- `pyyaml` is the only new dependency; add it to `requirements.txt`
- Existing behavior must be preserved: GDrive source still infers `program`, `district`, `academic_year`, `doc_type` -- those are now just default tag key names that a forker can rename via config
- `theme_cluster` stays as a special filter (derived by LLM, not a folder tag); it is NOT in `tags_config.yaml` `filter_keys`
- After Task 3, any existing `documents.parquet` / `themes.parquet` / Chroma DB must be rebuilt -- note this explicitly when finishing Task 3
- No changes to `pipeline/sources/local.py` or `pipeline/sources/base.py` -- the local source is already generic
- Do not rename Python variables `program`, `district`, `academic_year` inside `get_docs.py` logic -- only their source (config vs hardcode) changes
- Run `pytest tests/` after each task before committing

---

## File Map

| File | Action | What changes |
|---|---|---|
| `tags_config.yaml` | **Create** | Single source of truth for org-specific values |
| `requirements.txt` | **Modify** | Add `pyyaml>=6.0` |
| `config.py` | **Modify** | Load `tags_config.yaml`; expose `GDRIVE_PROGRAM_MAP`, `GDRIVE_KNOWN_DISTRICTS`, `GDRIVE_DISTRICT_DISPLAY`, `FILTER_TAG_KEYS` |
| `pipeline/get_docs.py` | **Modify** | Remove `PROGRAM_MAP`, `KNOWN_DISTRICTS`, `DISTRICT_DISPLAY` constants; import from `config` |
| `pipeline/ingest.py` | **Modify** | Write `tags` JSON column instead of individual `program`/`district`/etc. columns |
| `pipeline/build_vectorstore.py` | **Modify** | Flatten `tags` JSON into Chroma metadata; `_build_prefix` reads from `tags` |
| `pipeline/extract_themes.py` | **Modify** | `CARRY_COLS` uses `tags`; district inference reads `GDRIVE_KNOWN_DISTRICTS` from config |
| `pipeline/deduplicate_themes.py` | **Modify** | Drop hardcoded `district` column references; handle `tags` JSON column |
| `app/rag_pipeline.py` | **Modify** | `answer()` takes `tag_filters: dict`; `_build_where` is generic |
| `app/agent.py` | **Modify** | `chat()` takes `tag_filters: dict`; filter summary line built from dict |
| `app/tools.py` | **Modify** | Build `tag_filters` dict from named params; update `_build_where` calls |
| `app/app.py` | **Modify** | Sidebar loops over `FILTER_TAG_KEYS`; `get_themes()` expands `tags` JSON into columns |
| `tests/test_config.py` | **Modify** | Add tests for `tags_config.yaml` loading |
| `tests/test_ingest_tags.py` | **Create** | Tests for new `tags` JSON column in ingest output |
| `tests/test_rag_filters.py` | **Create** | Tests for generic `_build_where(tag_filters, theme_cluster)` |

---
### Task 1: `tags_config.yaml` + `config.py` + `requirements.txt`

**Files:**
- Create: `tags_config.yaml`
- Modify: `requirements.txt`
- Modify: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.GDRIVE_PROGRAM_MAP: dict[str, str]`, `config.GDRIVE_KNOWN_DISTRICTS: list[str]`, `config.GDRIVE_DISTRICT_DISPLAY: dict[str, str]`, `config.FILTER_TAG_KEYS: list[str]`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_config.py`:

```python
def test_tags_config_missing_file_gives_empty_defaults(monkeypatch, tmp_path):
    """config loads gracefully when tags_config.yaml does not exist."""
    import importlib
    import config
    monkeypatch.setattr(config, "_TAGS_CONFIG_PATH", tmp_path / "nonexistent.yaml")
    importlib.reload(config)
    assert config.GDRIVE_PROGRAM_MAP == {}
    assert config.GDRIVE_KNOWN_DISTRICTS == []
    assert config.GDRIVE_DISTRICT_DISPLAY == {}
    assert config.FILTER_TAG_KEYS == []


def test_tags_config_loads_values(monkeypatch, tmp_path):
    """config reads program_map, known_districts, district_display, filter_keys."""
    import importlib
    import yaml
    import config
    cfg = {
        "program_map": {"0_demo": "Demo Program"},
        "known_districts": ["alpha", "beta"],
        "district_display": {"alpha": "Alpha Region"},
        "filter_keys": ["program", "district"],
    }
    cfg_path = tmp_path / "tags_config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    monkeypatch.setattr(config, "_TAGS_CONFIG_PATH", cfg_path)
    importlib.reload(config)
    assert config.GDRIVE_PROGRAM_MAP == {"0_demo": "Demo Program"}
    assert config.GDRIVE_KNOWN_DISTRICTS == ["alpha", "beta"]
    assert config.GDRIVE_DISTRICT_DISPLAY == {"alpha": "Alpha Region"}
    assert config.FILTER_TAG_KEYS == ["program", "district"]
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_config.py::test_tags_config_missing_file_gives_empty_defaults tests/test_config.py::test_tags_config_loads_values -v
```

Expected: `AttributeError: module 'config' has no attribute 'GDRIVE_PROGRAM_MAP'`

- [ ] **Step 3: Add `pyyaml` to `requirements.txt`**

Add this line (after `python-dotenv`):

```
pyyaml>=6.0
```

- [ ] **Step 4: Create `tags_config.yaml` at the project root**

```yaml
# tags_config.yaml
# Customize this file for your document library. It controls how metadata is
# inferred from Google Drive folder paths and which filters appear in the app.
#
# For LOCAL files (SOURCE_TYPE=local), see FOLDER_METADATA_LEVELS in .env --
# that setting maps folder depth to tag key without any values here.

# Maps lowercase GDrive top-level folder names to display names.
program_map:
  "!_multiple programs": "Multiple Programs"
  "0_ekumen outreach": "Ekumen Outreach"
  "1_ansible studies": "Ansible Studies"
  "2_hainish mathematics": "Hainish Mathematics"
  "3_mobile training": "Mobile Training"
  "4_odonian method": "Odonian Method"
  "4_odonian": "Odonian Method"
  "5_ansible initiative": "Ansible Initiative"
  "6_ekumen council": "Ekumen Council"
  "background": "Background"

# Substring list for inferring the "district" tag from folder paths and file names.
# Delete this section entirely if your documents are not organized by region.
known_districts:
  - "rocannon's world"
  - "gethen"
  - "anarres"
  - "urras"
  - "athshe"
  - "werel"
  - "yeowe"
  - "seggri"
  - "aka"
  - "davenant"
  - "chiffewar"
  - "hain"
  - "terra"
  - "karhide"
  - "orgoreyn"
  - "abbenay"
  - "a-io"
  - "ekumen central"

# Display-name overrides for district values where .title() gives wrong results.
district_display:
  "rocannon's world": "Rocannon's World"
  "a-io": "A-Io"
  "ekumen central": "Ekumen Central"

# Tag keys that appear as filter dropdowns in the app sidebar.
filter_keys:
  - program
  - district
  - academic_year
  - doc_type
```

- [ ] **Step 5: Update `config.py`**

Add this block after the `CHROMA_COLLECTION` line:

```python
# ------------------------------------------------------------------
# Tags config  (drives GDrive folder inference + app sidebar filters)
# ------------------------------------------------------------------
try:
    import yaml as _yaml

    def _load_tags_config(path: "Path") -> dict:
        if path.exists():
            with open(path, encoding="utf-8") as _f:
                return _yaml.safe_load(_f) or {}
        return {}

except ImportError:
    def _load_tags_config(path: "Path") -> dict:  # type: ignore[misc]
        return {}

_TAGS_CONFIG_PATH = PROJECT_ROOT / "tags_config.yaml"
_TAGS_CONFIG = _load_tags_config(_TAGS_CONFIG_PATH)

GDRIVE_PROGRAM_MAP: dict[str, str]      = _TAGS_CONFIG.get("program_map", {})
GDRIVE_KNOWN_DISTRICTS: list[str]       = _TAGS_CONFIG.get("known_districts", [])
GDRIVE_DISTRICT_DISPLAY: dict[str, str] = _TAGS_CONFIG.get("district_display", {})
FILTER_TAG_KEYS: list[str]              = _TAGS_CONFIG.get("filter_keys", [])
```

- [ ] **Step 6: Run tests to confirm they pass**

```
pytest tests/test_config.py -v
```

Expected: all tests PASS (including the two new ones).

- [ ] **Step 7: Commit**

```bash
git add tags_config.yaml requirements.txt config.py tests/test_config.py
git commit -m "feat: load tag mappings from tags_config.yaml"
```

---

### Task 2: `get_docs.py` -- config-driven inference

**Files:**
- Modify: `pipeline/get_docs.py`

**Interfaces:**
- Consumes: `config.GDRIVE_PROGRAM_MAP`, `config.GDRIVE_KNOWN_DISTRICTS`, `config.GDRIVE_DISTRICT_DISPLAY`
- Produces: same `metadata.json` output as before -- no output schema change in this task

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py` (or create `tests/test_get_docs.py`):

```python
def test_get_docs_reads_program_map_from_config(monkeypatch, tmp_path):
    """get_docs.PROGRAM_MAP is populated from config, not hardcoded."""
    import importlib, yaml, config, pipeline.get_docs as gd
    cfg = {
        "program_map": {"0_test program": "Test Program"},
        "known_districts": [],
        "district_display": {},
        "filter_keys": ["program"],
    }
    cfg_path = tmp_path / "tags_config.yaml"
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    monkeypatch.setattr(config, "_TAGS_CONFIG_PATH", cfg_path)
    importlib.reload(config)
    importlib.reload(gd)
    assert gd.PROGRAM_MAP == {"0_test program": "Test Program"}
```

- [ ] **Step 2: Run test to confirm it fails**

```
pytest tests/test_config.py::test_get_docs_reads_program_map_from_config -v
```

Expected: FAIL -- `PROGRAM_MAP` is still hardcoded.

- [ ] **Step 3: Update `pipeline/get_docs.py`**

At the top of the file (after existing imports), add:

```python
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import config as _config
```

Then remove the three hardcoded constant blocks entirely and replace with:

```python
# Loaded from tags_config.yaml via config.py
PROGRAM_MAP:      dict[str, str] = _config.GDRIVE_PROGRAM_MAP
KNOWN_DISTRICTS:  list[str]      = _config.GDRIVE_KNOWN_DISTRICTS
DISTRICT_DISPLAY: dict[str, str] = {d: d.title() for d in KNOWN_DISTRICTS}
DISTRICT_DISPLAY.update(_config.GDRIVE_DISTRICT_DISPLAY)
```

- [ ] **Step 4: Run full tests**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add pipeline/get_docs.py tests/test_config.py
git commit -m "feat: get_docs reads program/district lists from config"
```

---

### Task 3: Tags-based pipeline schema

**Files:**
- Modify: `pipeline/ingest.py`
- Modify: `pipeline/build_vectorstore.py`
- Modify: `pipeline/extract_themes.py`
- Modify: `pipeline/deduplicate_themes.py`
- Create: `tests/test_ingest_tags.py`

**Interfaces:**
- Consumes: `metadata.json` records with `program`, `district`, `academic_year`, etc. as top-level fields
- Produces: `documents.parquet` with a `tags` JSON string column (e.g. `'{"program": "Ekumen Outreach", "district": "Gethen"}'`) instead of individual named columns

> WARNING: After this task, any existing `documents.parquet`, `themes_raw.parquet`, `themes.parquet`, and Chroma DB are schema-incompatible. Delete them and rebuild by re-running the full pipeline.

- [ ] **Step 1: Write failing tests in `tests/test_ingest_tags.py`**

```python
import json
import pytest
from pathlib import Path


def test_ingest_main_writes_tags_json_column(tmp_path, monkeypatch):
    """ingest.main() writes a 'tags' JSON column, not individual program/district columns."""
    import json as _json
    from pipeline import ingest

    metadata = [{
        "file_id":         "abc123",
        "file_name":       "report.pdf",
        "mime_type":       "application/pdf",
        "folder_path":     "0_ekumen outreach",
        "local_path":      str(tmp_path / "report.pdf"),
        "drive_url":       "",
        "download_status": "exists",
        "program":         "Ekumen Outreach",
        "district":        "Gethen",
        "academic_year":   "2024-25",
        "doc_type":        "site_visit",
        "season":          "Fall",
        "date_precision":  "direct",
    }]
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(_json.dumps(metadata), encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"fake pdf content")

    out_path = tmp_path / "documents.parquet"
    monkeypatch.setattr(ingest, "METADATA_PATH", meta_path)
    monkeypatch.setattr(ingest, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    ingest.main()

    import pandas as pd
    df = pd.read_parquet(out_path)
    assert "tags" in df.columns, "Expected 'tags' column in output"
    assert "program" not in df.columns, "Expected 'program' column to be removed"
    assert "district" not in df.columns, "Expected 'district' column to be removed"
    tags = _json.loads(df.iloc[0]["tags"])
    assert tags.get("program") == "Ekumen Outreach"
    assert tags.get("district") == "Gethen"


def test_ingest_tags_excludes_system_fields(tmp_path, monkeypatch):
    """tags column does not include file_id, file_name, mime_type, etc."""
    import json as _json
    from pipeline import ingest

    metadata = [{
        "file_id":         "x1",
        "file_name":       "f.pdf",
        "mime_type":       "application/pdf",
        "folder_path":     "",
        "local_path":      str(tmp_path / "f.pdf"),
        "drive_url":       "",
        "download_status": "exists",
        "program":         "Ansible Studies",
    }]
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(_json.dumps(metadata), encoding="utf-8")
    (tmp_path / "f.pdf").write_bytes(b"x")

    out_path = tmp_path / "documents.parquet"
    monkeypatch.setattr(ingest, "METADATA_PATH", meta_path)
    monkeypatch.setattr(ingest, "OUTPUT_PATH", out_path)
    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    ingest.main()

    import pandas as pd
    df = pd.read_parquet(out_path)
    tags = _json.loads(df.iloc[0]["tags"])
    for field in ("file_id", "file_name", "mime_type", "drive_url", "folder_path"):
        assert field not in tags, f"System field '{field}' should not be in tags"
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_ingest_tags.py -v
```

Expected: FAIL -- output has `program` column, no `tags` column.

- [ ] **Step 3: Update `pipeline/ingest.py`**

Near the top of the file (after imports), add:

```python
# Fields that are NOT tags -- they are structural/identity fields
_SYSTEM_FIELDS = frozenset({
    "file_id", "file_name", "mime_type", "folder_path",
    "local_path", "drive_url", "download_status",
})
```

In `main()`, find the `rows.append({...})` block and replace it:

```python
_tags = {
    k: v for k, v in record.items()
    if k not in _SYSTEM_FIELDS and v is not None and str(v).strip()
}
rows.append({
    "file_id":           record.get("file_id"),
    "file_name":         record.get("file_name"),
    "mime_type":         record.get("mime_type"),
    "folder_path":       record.get("folder_path"),
    "local_path":        record.get("local_path"),
    "drive_url":         record.get("drive_url"),
    "tags":              json.dumps(_tags, ensure_ascii=False),
    "text":              text,
    "headings":          json.dumps(headings, ensure_ascii=False),
    "char_count":        len(text),
    "extraction_status": status,
    "extraction_error":  error,
})
```

Apply the identical change to the `rows.append` inside `process_stream()`.

- [ ] **Step 4: Update `pipeline/build_vectorstore.py`**

Replace `_build_prefix`:

```python
def _build_prefix(rec: dict, section: str | None) -> str:
    """Contextual header prepended to each chunk before embedding."""
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
```

In `embed_and_index`, replace the metadata construction block:

```python
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
```

- [ ] **Step 5: Update `pipeline/extract_themes.py`**

Replace `CARRY_COLS`:

```python
CARRY_COLS = [
    "file_id", "file_name", "mime_type", "folder_path", "local_path", "drive_url",
    "tags",   # replaces individual program/doc_type/academic_year/season/date_precision/district
    "char_count", "extraction_status",
]
```

Add import at top:

```python
from config import GDRIVE_KNOWN_DISTRICTS
```

Remove the hardcoded `KNOWN_DISTRICTS` constant and replace with:

```python
KNOWN_DISTRICTS: list[str] = GDRIVE_KNOWN_DISTRICTS
```

Find the `needs_district` logic (currently checks `df["district"].isna()`). Replace with:

```python
try:
    existing_tags: dict = json.loads(row.get("tags") or "{}")
except (json.JSONDecodeError, TypeError):
    existing_tags = {}
needs_district = bool(KNOWN_DISTRICTS) and not existing_tags.get("district")
```

In the section where inferred fields are written back, replace individual column writes with:

```python
updated_tags = dict(existing_tags)
if fields.get("inferred_district"):
    updated_tags["district"] = fields["inferred_district"]
# In the output row dict:
"tags": json.dumps(updated_tags, ensure_ascii=False),
```

- [ ] **Step 6: Update `pipeline/deduplicate_themes.py`**

Find and remove the inferred-district promotion block entirely:

```python
# DELETE:
if "inferred_district" in out.columns:
    dist_mask = out["district"].isna() & out["inferred_district"].notna()
    if dist_mask.any():
        out.loc[dist_mask, "district"] = out.loc[dist_mask, "inferred_district"]
        print(f"  Promoted inferred district for {dist_mask.sum()} documents")
```

For any remaining references to `out["district"]` or `out["program"]` as standalone columns, add a helper and use it:

```python
import json as _json

def _get_tag(row: dict, key: str) -> "str | None":
    try:
        return _json.loads(row.get("tags") or "{}").get(key)
    except (json.JSONDecodeError, TypeError):
        return None
```

- [ ] **Step 7: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit**

```bash
git add pipeline/ingest.py pipeline/build_vectorstore.py pipeline/extract_themes.py pipeline/deduplicate_themes.py tests/test_ingest_tags.py
git commit -m "feat: pipeline stores tags as JSON column, vectorstore flattens at index time"
```

> REBUILD REQUIRED: Delete `data/documents.parquet`, `data/themes_raw.parquet`, `data/themes.parquet`, and `data/chroma_db/` then re-run:
>
> ```
> python pipeline/get_docs.py
> python pipeline/ingest.py
> python pipeline/extract_themes.py
> python pipeline/deduplicate_themes.py
> python pipeline/build_vectorstore.py
> ```

---

### Task 4: Generic filter interface

**Files:**
- Modify: `app/rag_pipeline.py`
- Modify: `app/agent.py`
- Modify: `app/tools.py`
- Modify: `app/app.py`
- Create: `tests/test_rag_filters.py`

**Interfaces:**
- `RagPipeline.answer(question, *, tag_filters: dict[str,str|None]|None, theme_cluster: str|None, history: list|None) -> dict`
- `_build_where(tag_filters: dict[str,str|None]|None, theme_cluster: str|None) -> dict|None`
- `Agent.chat(question, *, tag_filters: dict[str,str|None]|None, theme_cluster: str|None, history: list|None) -> dict`

- [ ] **Step 1: Write failing tests in `tests/test_rag_filters.py`**

```python
def test_build_where_returns_none_for_empty_filters():
    from app.rag_pipeline import _build_where
    assert _build_where(None, None) is None


def test_build_where_single_tag():
    from app.rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach"}, None)
    assert result == {"program": {"$eq": "Ekumen Outreach"}}


def test_build_where_multiple_tags():
    from app.rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach", "district": "Gethen"}, None)
    assert result == {
        "$and": [
            {"program":  {"$eq": "Ekumen Outreach"}},
            {"district": {"$eq": "Gethen"}},
        ]
    }


def test_build_where_ignores_none_values():
    from app.rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach", "district": None}, None)
    assert result == {"program": {"$eq": "Ekumen Outreach"}}


def test_build_where_theme_cluster():
    from app.rag_pipeline import _build_where
    result = _build_where(None, "Institutional Change")
    assert result == {"theme_clusters": {"$contains": "Institutional Change"}}
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_rag_filters.py -v
```

Expected: FAIL -- `_build_where` has old signature `(program, district, academic_year, doc_type, theme_cluster)`.

- [ ] **Step 3: Replace `_build_where` in `app/rag_pipeline.py`**

```python
def _build_where(
    tag_filters: "dict[str, str | None] | None",
    theme_cluster: "str | None",
) -> "dict | None":
    conditions: list[dict] = []
    for key, val in (tag_filters or {}).items():
        if val:
            conditions.append({key: {"$eq": val}})
    if theme_cluster:
        conditions.append({"theme_clusters": {"$contains": theme_cluster}})
    if not conditions:
        return None
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}
```

- [ ] **Step 4: Replace `RagPipeline.answer()` signature and body**

```python
def answer(
    self,
    question: str,
    *,
    tag_filters: "dict[str, str | None] | None" = None,
    theme_cluster: "str | None" = None,
    history: "list[dict] | None" = None,
) -> dict:
    where  = _build_where(tag_filters, theme_cluster)
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

    messages    = _build_messages(question, chunks, theme_ctx, history)
    response    = self._llm.invoke(_to_lc_messages(messages))
    answer_text = response.content.strip()

    sources = [
        {
            "n":         c["n"],
            "file_name": c["meta"].get("file_name", ""),
            "drive_url": c["meta"].get("drive_url", ""),
            "section":   _section_label(c["meta"]),
            "text":      c["text"],
            **{
                k: c["meta"].get(k, "")
                for k in c["meta"]
                if k not in {
                    "file_id", "file_name", "drive_url", "folder_path",
                    "section_h1", "section_h2", "section_h3",
                    "chunk_index", "chunk_count",
                }
            },
        }
        for c in chunks
    ]

    return {"answer": answer_text, "sources": sources}
```

- [ ] **Step 5: Run filter tests**

```
pytest tests/test_rag_filters.py -v
```

Expected: all five tests PASS.

- [ ] **Step 6: Update `app/agent.py` -- `chat()` signature**

Replace:

```python
def chat(
    self, question: str, *,
    history: list[dict] | None = None,
    program: str | None = None,
    district: str | None = None,
    academic_year: str | None = None,
    doc_type: str | None = None,
    theme_cluster: str | None = None,
) -> dict:
    ...
    filter_lines = []
    if program:       filter_lines.append(f"Program: {program}")
    if district:      filter_lines.append(f"District: {district}")
    if academic_year: filter_lines.append(f"Year: {academic_year}")
    if doc_type:      filter_lines.append(f"Doc type: {doc_type}")
    if theme_cluster: filter_lines.append(f"Theme cluster: {theme_cluster}")
```

With:

```python
def chat(
    self, question: str, *,
    history: list[dict] | None = None,
    tag_filters: "dict[str, str | None] | None" = None,
    theme_cluster: str | None = None,
) -> dict:
    ...
    filter_lines = [
        f"{k.replace('_', ' ').title()}: {v}"
        for k, v in (tag_filters or {}).items()
        if v
    ]
    if theme_cluster:
        filter_lines.append(f"Theme cluster: {theme_cluster}")
```

Update the CLI argument parser at the bottom of `agent.py`:

```python
parser.add_argument(
    "--filter", action="append", default=[], metavar="KEY=VALUE",
    help="Tag filter, e.g. --filter 'program=Ekumen Outreach'",
)
parser.add_argument("--theme-cluster", default=None, dest="theme_cluster")
args = parser.parse_args()

tag_filters = {}
for f in args.filter:
    if "=" in f:
        k, v = f.split("=", 1)
        tag_filters[k.strip()] = v.strip().strip("'\"")

result = ag.chat(
    args.question,
    tag_filters=tag_filters or None,
    theme_cluster=args.theme_cluster,
)
```

- [ ] **Step 7: Update `app/tools.py` -- all `_build_where` and `pipeline.answer` call sites**

There are 5 `_build_where` call sites (one per tool function). For each:

```python
# REMOVE:
where = _build_where(program, district, academic_year, doc_type, theme_cluster)

# REPLACE WITH:
_tag_filters = {k: v for k, v in {
    "program": program, "district": district,
    "academic_year": academic_year, "doc_type": doc_type,
}.items() if v}
where = _build_where(_tag_filters or None, theme_cluster)
```

For each `pipeline.answer(...)` call:

```python
# REMOVE:
result = pipeline.answer(
    query, program=program, district=district,
    academic_year=academic_year, doc_type=doc_type, theme_cluster=theme_cluster,
)

# REPLACE WITH:
_tag_filters = {k: v for k, v in {
    "program": program, "district": district,
    "academic_year": academic_year, "doc_type": doc_type,
}.items() if v}
result = pipeline.answer(
    query,
    tag_filters=_tag_filters or None,
    theme_cluster=theme_cluster,
)
```

Note: `tools.py` keeps its explicit named params so the LLM sees descriptive parameter names. The dict-building happens inside each function.

- [ ] **Step 8: Update `app/app.py` -- dynamic sidebar + tags expansion**

Add at top of `app.py`:

```python
import config as _config
```

Replace `get_themes()` to expand the `tags` JSON column into individual columns:

```python
@st.cache_data(show_spinner=False)
def get_themes() -> pd.DataFrame | None:
    if not THEMES_PATH.exists():
        return None
    df = pd.read_parquet(THEMES_PATH)
    if "theme_extraction_status" in df.columns:
        df = df[df["theme_extraction_status"] == "ok"]
    if "tags" in df.columns:
        import json as _json

        def _parse_tags(val) -> dict:
            if not val:
                return {}
            try:
                return _json.loads(val) if isinstance(val, str) else (val or {})
            except Exception:
                return {}

        tags_expanded = df["tags"].apply(_parse_tags).apply(pd.Series)
        df = df.drop(columns=["tags"]).join(tags_expanded)
    return df if not df.empty else None
```

Replace the hardcoded Program/District/Academic Year/Doc Type sidebar blocks with a loop:

```python
filters: dict = {}
for _key in _config.FILTER_TAG_KEYS:
    _label   = _key.replace("_", " ").title()
    _options = _distinct(themes_df, _key)
    if not _options:
        continue
    st.markdown(
        f"<div class='sidebar-field-label'>{_label}</div>",
        unsafe_allow_html=True,
    )
    _val = st.selectbox(
        "", ["(All)"] + _options,
        key=_key,
        label_visibility="collapsed",
    )
    filters[_key] = None if _val == "(All)" else _val

# Theme cluster stays as a special filter
st.markdown(
    "<div class='sidebar-field-label'>Theme Cluster</div>",
    unsafe_allow_html=True,
)
_tc_options = _distinct_clusters(themes_df)
if _tc_options:
    _tc_val = st.selectbox(
        "", ["(All)"] + _tc_options, key="theme_cluster", label_visibility="collapsed"
    )
    filters["theme_cluster"] = None if _tc_val == "(All)" else _tc_val
else:
    filters["theme_cluster"] = None
```

Update `sidebar()` return:

```python
return filters
```

Update where the sidebar result is used to call `agent.chat()`:

```python
_theme_cluster = filters.pop("theme_cluster", None)
result = _agent.chat(
    question,
    tag_filters={k: v for k, v in filters.items() if v} or None,
    theme_cluster=_theme_cluster,
    history=history,
)
```

- [ ] **Step 9: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests PASS.

- [ ] **Step 10: Commit**

```bash
git add app/rag_pipeline.py app/agent.py app/tools.py app/app.py tests/test_rag_filters.py
git commit -m "feat: generic tag_filters interface replaces hardcoded program/district params"
```

---

## Post-implementation checklist

- [ ] `tags_config.yaml` is committed to the repo (it contains no secrets -- just display names)
- [ ] `README.md` points to `tags_config.yaml` as the first file to customize when forking
- [ ] Rebuild the full data pipeline on real data to confirm end-to-end correctness
- [ ] Confirm the Streamlit sidebar reflects `filter_keys` from `tags_config.yaml`
