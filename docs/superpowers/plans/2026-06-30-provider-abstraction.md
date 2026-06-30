# Provider Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make docs-to-rag-chat provider-agnostic -- swap LLM providers (OpenAI, Anthropic, Gemini, Ollama) and document sources (Google Drive, local folder) via env vars, with Ollama local embeddings auto-selected when the chat provider is Anthropic or Ollama.

**Architecture:** A `config.py` at the project root centralises every env var; a `providers.py` at the project root exposes two factory functions (`build_chat_model`, `build_embedder`) using LangChain unified `BaseChatModel` / `Embeddings` interfaces; `pipeline/sources/` adds a `DocumentSource` Protocol with `GoogleDriveSource` and `LocalFolderSource` implementations. All existing pipeline scripts and app modules are updated to call the factories instead of hardcoding OpenAI.

**Tech Stack:** LangChain core (already installed), `langchain-anthropic`, `langchain-google-genai`, `langchain-ollama`, existing Chroma + Streamlit stack.

## Global Constraints

- Python 3.11+
- All new code must be importable from both `app/` (Streamlit) and `pipeline/` (CLI) contexts -- each adds its own directory to `sys.path`. Handle this by inserting `PROJECT_ROOT` into `sys.path` before importing `config` or `providers`.
- No new mandatory runtime deps -- provider packages are optional; import them inside the matching `if` branch (late imports).
- `response_format={"type": "json_object"}` is OpenAI-specific -- drop it. All existing LLM prompts already instruct JSON-only output.
- Keep `get_docs.py` and its `stream_docs` / `build_drive_service` functions intact -- `GoogleDriveSource` wraps them rather than duplicating.
- `langchain-openai` is already in `requirements.txt`. Do not remove it.
- Tests live in `tests/` at the project root. Run with `pytest tests/ -v`.

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `config.py` | All env vars; embedding auto-selection logic |
| Create | `providers.py` | `build_chat_model()` and `build_embedder()` factories |
| Create | `pipeline/sources/__init__.py` | Module marker |
| Create | `pipeline/sources/base.py` | `DocumentRecord` TypedDict + `DocumentSource` Protocol |
| Create | `pipeline/sources/gdrive.py` | `GoogleDriveSource` wrapping `get_docs.py` |
| Create | `pipeline/sources/local.py` | `LocalFolderSource` walking a local directory |
| Create | `.env.example` | Documented env-var template |
| Create | `tests/__init__.py` | Module marker |
| Create | `tests/test_config.py` | Config defaults and auto-selection |
| Create | `tests/test_providers.py` | Factory return types |
| Create | `tests/test_sources.py` | Local source with real filesystem |
| Modify | `app/rag_pipeline.py` | Use `build_embedder()` + `build_chat_model()` |
| Modify | `app/agent.py` | Use `build_chat_model()` |
| Modify | `app/tools.py` | Use `build_chat_model()` in `synthesize` tool |
| Modify | `pipeline/build_vectorstore.py` | Use `build_embedder()`; fix `COLLECTION` constant; fix `do_rebuild` bug |
| Modify | `pipeline/extract_themes.py` | Use `build_chat_model()` |
| Modify | `pipeline/deduplicate_themes.py` | Use `build_chat_model()` |
| Modify | `pipeline/__init__.py` | Use source abstraction for Step 2 |
| Modify | `requirements.txt` | Add new provider packages |

---

## Task 1: config.py + .env.example

**Files:**
- Create: `config.py`
- Create: `.env.example`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Produces: `config.LLM_PROVIDER`, `config.LLM_MODEL`, `config.LLM_TEMPERATURE`, `config.EMBED_PROVIDER`, `config.EMBED_MODEL`, `config.OLLAMA_BASE_URL`, `config.SOURCE_TYPE`, `config.LOCAL_DOCS_DIR`, `config.SHARED_DRIVE_ID`, `config.CHROMA_COLLECTION`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
import importlib
import os
import pytest


def _reload(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    importlib.reload(config)
    return config


def test_defaults(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("EMBED_PROVIDER", raising=False)
    monkeypatch.delenv("EMBED_MODEL", raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.LLM_PROVIDER == "openai"
    assert cfg.EMBED_PROVIDER == "openai"
    assert cfg.EMBED_MODEL == "text-embedding-3-small"


def test_anthropic_auto_selects_ollama_embed(monkeypatch):
    cfg = _reload(monkeypatch, LLM_PROVIDER="anthropic")
    assert cfg.EMBED_PROVIDER == "ollama"
    assert cfg.EMBED_MODEL == "nomic-embed-text"


def test_ollama_auto_selects_ollama_embed(monkeypatch):
    cfg = _reload(monkeypatch, LLM_PROVIDER="ollama")
    assert cfg.EMBED_PROVIDER == "ollama"


def test_google_auto_selects_google_embed(monkeypatch):
    cfg = _reload(monkeypatch, LLM_PROVIDER="google")
    assert cfg.EMBED_PROVIDER == "google"
    assert cfg.EMBED_MODEL == "models/text-embedding-004"


def test_embed_provider_can_be_overridden(monkeypatch):
    cfg = _reload(monkeypatch, LLM_PROVIDER="anthropic", EMBED_PROVIDER="openai")
    assert cfg.EMBED_PROVIDER == "openai"


def test_source_type_default(monkeypatch):
    monkeypatch.delenv("SOURCE_TYPE", raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.SOURCE_TYPE == "gdrive"


def test_chroma_collection_default(monkeypatch):
    monkeypatch.delenv("CHROMA_COLLECTION", raising=False)
    cfg = _reload(monkeypatch)
    assert cfg.CHROMA_COLLECTION == "docs"
```

- [ ] **Step 2: Run tests -- verify they fail**

```
pytest tests/test_config.py -v
```

Expected: `ModuleNotFoundError: No module named config`

- [ ] **Step 3: Create config.py**

```python
# config.py  (project root)
"""
Central configuration -- reads from secrets/env-config (rename to .env).
Every configurable value lives here; nothing else should read env vars directly.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
load_dotenv(PROJECT_ROOT / "secrets" / ".env")

# ------------------------------------------------------------------
# LLM
# ------------------------------------------------------------------
LLM_PROVIDER    = os.getenv("LLM_PROVIDER",    "openai").lower()   # openai|anthropic|google|ollama
LLM_MODEL       = os.getenv("LLM_MODEL",       "gpt-4o-mini")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# ------------------------------------------------------------------
# Embeddings
# Anthropic and Ollama both fall back to local Ollama embeddings
# because Anthropic offers no embedding model.
# Override with EMBED_PROVIDER / EMBED_MODEL env vars if needed.
# ------------------------------------------------------------------
_EMBED_PROVIDER_DEFAULTS: dict[str, str] = {
    "openai":    "openai",
    "google":    "google",
    "anthropic": "ollama",
    "ollama":    "ollama",
}
_EMBED_MODEL_DEFAULTS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "google": "models/text-embedding-004",
    "ollama": "nomic-embed-text",
}

EMBED_PROVIDER = (
    os.getenv("EMBED_PROVIDER") or _EMBED_PROVIDER_DEFAULTS.get(LLM_PROVIDER, "openai")
).lower()
EMBED_MODEL = (
    os.getenv("EMBED_MODEL") or _EMBED_MODEL_DEFAULTS.get(EMBED_PROVIDER, "text-embedding-3-small")
)

# ------------------------------------------------------------------
# Ollama  (used when LLM_PROVIDER or EMBED_PROVIDER is "ollama")
# ------------------------------------------------------------------
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# ------------------------------------------------------------------
# Document source
# ------------------------------------------------------------------
SOURCE_TYPE    = os.getenv("SOURCE_TYPE", "gdrive").lower()   # gdrive | local
LOCAL_DOCS_DIR = os.getenv("LOCAL_DOCS_DIR", "")

# ------------------------------------------------------------------
# Google Drive  (only needed when SOURCE_TYPE=gdrive)
# ------------------------------------------------------------------
SHARED_DRIVE_ID = os.getenv("SHARED_DRIVE_ID", "")

# ------------------------------------------------------------------
# Vector store
# ------------------------------------------------------------------
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")
```

- [ ] **Step 4: Create tests/__init__.py (empty file)**

- [ ] **Step 5: Run tests -- verify they pass**

```
pytest tests/test_config.py -v
```

Expected: 7 passed

- [ ] **Step 6: Create .env.example**

```bash
# .env.example -- copy to secrets/.env and fill in your values.
# The secrets/ directory is gitignored; never commit real credentials.

# ------------------------------------------------------------------
# LLM provider -- pick one
# ------------------------------------------------------------------
LLM_PROVIDER=openai                        # openai | anthropic | google | ollama
LLM_MODEL=gpt-4o-mini                      # model name for the chosen provider
LLM_TEMPERATURE=0.1

# OpenAI
OPENAI_API_KEY=<your-openai-api-key>

# Anthropic (set LLM_PROVIDER=anthropic)
# ANTHROPIC_API_KEY=<your-anthropic-api-key>

# Google Gemini (set LLM_PROVIDER=google)
# GOOGLE_API_KEY=<your-google-api-key>

# Ollama (set LLM_PROVIDER=ollama -- runs fully locally, no API key needed)
# OLLAMA_BASE_URL=http://localhost:11434
# LLM_MODEL=llama3.2

# ------------------------------------------------------------------
# Embeddings
# Auto-selected from LLM_PROVIDER. Override only if you want
# a different embedding provider than the default for your LLM.
# Note: changing EMBED_MODEL requires rebuilding data/chroma_db/.
# ------------------------------------------------------------------
# EMBED_PROVIDER=openai                    # openai | google | ollama
# EMBED_MODEL=text-embedding-3-small

# ------------------------------------------------------------------
# Document source
# ------------------------------------------------------------------
SOURCE_TYPE=gdrive                         # gdrive | local

# Google Drive (SOURCE_TYPE=gdrive)
SHARED_DRIVE_ID=your-shared-drive-id-here

# Local folder (SOURCE_TYPE=local)
# LOCAL_DOCS_DIR=/path/to/your/documents

# ------------------------------------------------------------------
# Google service account (Drive access and feedback sheet)
# ------------------------------------------------------------------
GOOGLE_SERVICE_ACCOUNT_KEY=secrets/google-service-account.json

# ------------------------------------------------------------------
# Feedback (optional -- comment out to disable feedback collection)
# ------------------------------------------------------------------
# FEEDBACK_SHEET_ID=your-google-sheet-id
# FEEDBACK_TAB_GID=0

# ------------------------------------------------------------------
# Vector store (rarely needs changing)
# ------------------------------------------------------------------
# CHROMA_COLLECTION=docs
```

- [ ] **Step 7: Commit**

```bash
git add config.py .env.example tests/__init__.py tests/test_config.py
git commit -m "feat: add central config.py with provider and source env vars"
```

---

## Task 2: providers.py -- chat model + embedder factories

**Files:**
- Create: `providers.py`
- Create: `tests/test_providers.py`

**Interfaces:**
- Consumes: `config.LLM_PROVIDER`, `config.LLM_MODEL`, `config.LLM_TEMPERATURE`, `config.EMBED_PROVIDER`, `config.EMBED_MODEL`, `config.OLLAMA_BASE_URL`
- Produces:
  - `build_chat_model() -> langchain_core.language_models.BaseChatModel`
  - `build_embedder() -> langchain_core.embeddings.Embeddings`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_providers.py
import importlib
import pytest
from unittest.mock import patch, MagicMock


def _reload_providers(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    importlib.reload(config)
    import providers
    importlib.reload(providers)
    return providers


def test_build_chat_model_openai(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai", LLM_MODEL="gpt-4o-mini")
    from langchain_openai import ChatOpenAI
    model = p.build_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_build_chat_model_anthropic(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="anthropic", LLM_MODEL="claude-haiku-4-5-20251001")
    from langchain_anthropic import ChatAnthropic
    model = p.build_chat_model()
    assert isinstance(model, ChatAnthropic)


def test_build_chat_model_google(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="google", LLM_MODEL="gemini-2.0-flash")
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = p.build_chat_model()
    assert isinstance(model, ChatGoogleGenerativeAI)


def test_build_chat_model_ollama(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="ollama", LLM_MODEL="llama3.2")
    from langchain_openai import ChatOpenAI
    model = p.build_chat_model()
    assert isinstance(model, ChatOpenAI)
    assert "11434" in (model.openai_api_base or "")


def test_build_chat_model_unknown_raises(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="unknown_provider")
    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        p.build_chat_model()


def test_build_embedder_openai(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai")
    from langchain_openai import OpenAIEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, OpenAIEmbeddings)


def test_build_embedder_ollama(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="anthropic")  # auto-selects ollama
    from langchain_ollama import OllamaEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, OllamaEmbeddings)


def test_build_embedder_google(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="google")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, GoogleGenerativeAIEmbeddings)


def test_build_embedder_unknown_raises(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai", EMBED_PROVIDER="unknown")
    with pytest.raises(ValueError, match="Unknown EMBED_PROVIDER"):
        p.build_embedder()
```

- [ ] **Step 2: Run tests -- verify they fail**

```
pytest tests/test_providers.py -v
```

Expected: `ModuleNotFoundError: No module named providers`

- [ ] **Step 3: Create providers.py**

```python
# providers.py  (project root)
"""
Provider factories -- returns LangChain-compatible models driven by config.py.

  build_chat_model()  ->  BaseChatModel  (openai / anthropic / google / ollama)
  build_embedder()    ->  Embeddings     (openai / google / ollama)

Provider packages are imported lazily so only the one in use needs to be installed.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


def build_chat_model() -> BaseChatModel:
    """Return the configured chat model as a LangChain BaseChatModel."""
    provider = config.LLM_PROVIDER
    model    = config.LLM_MODEL
    temp     = config.LLM_TEMPERATURE

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temp)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temp)  # type: ignore[call-arg]

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temp)

    if provider == "ollama":
        # Ollama exposes an OpenAI-compatible API endpoint
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            temperature=temp,
            base_url=f"{config.OLLAMA_BASE_URL}/v1",
            api_key="ollama",
        )

    raise ValueError(
        f"Unknown LLM_PROVIDER: {provider!r}. "
        "Choose one of: openai, anthropic, google, ollama"
    )


def build_embedder() -> Embeddings:
    """Return the configured embedding model as a LangChain Embeddings object."""
    provider = config.EMBED_PROVIDER
    model    = config.EMBED_MODEL

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model=model)

    if provider == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model=model)

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(model=model, base_url=config.OLLAMA_BASE_URL)

    raise ValueError(
        f"Unknown EMBED_PROVIDER: {provider!r}. "
        "Choose one of: openai, google, ollama"
    )
```

- [ ] **Step 4: Run tests -- verify they pass**

```
pytest tests/test_providers.py -v
```

Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add providers.py tests/test_providers.py
git commit -m "feat: add provider factories for chat model and embedder"
```

---

## Task 3: Wire providers into app/

**Files:**
- Modify: `app/rag_pipeline.py`
- Modify: `app/agent.py`
- Modify: `app/tools.py`

**Interfaces:**
- Consumes: `providers.build_chat_model()`, `providers.build_embedder()`

No automated tests for this task -- the app/ modules require a running Chroma DB and API keys to instantiate. Verify manually by running the Streamlit app after completing all steps.

- [ ] **Step 1: Update app/rag_pipeline.py**

Remove the hardcoded `OpenAI()` client. Add a `_to_lc_messages` helper and wire in both factories.

Replace the top-level constants block and imports (remove `from openai import OpenAI`, `EMBED_MODEL`, `CHAT_MODEL`, `TEMPERATURE` constants):

```python
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from providers import build_chat_model, build_embedder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import config as _config
COLLECTION = _config.CHROMA_COLLECTION    # replaces hardcoded "impact_florida_docs"
```

Replace `RagPipeline.__init__` -- remove `self.openai = OpenAI()`, add factories:

```python
def __init__(
    self,
    chroma_path: Path = CHROMA_PATH,
    themes_path: Path = THEMES_PATH,
) -> None:
    self._embedder = build_embedder()
    self._llm      = build_chat_model()

    chroma = chromadb.PersistentClient(path=str(chroma_path))
    try:
        self.collection = chroma.get_collection(COLLECTION)
    except Exception:
        raise RuntimeError(
            f"Chroma collection {COLLECTION!r} not found. "
            "Run pipeline/build_vectorstore.py first."
        )

    self.themes_df = None
    if themes_path.exists():
        self.themes_df = pd.read_parquet(
            themes_path,
            columns=["file_id", "themes", "key_findings", "theme_clusters"],
        )
```

Replace the embedding call inside `_retrieve`:

```python
query_vec = self._embedder.embed_query(question)
```

Add the helper just above the `RagPipeline` class definition:

```python
def _to_lc_messages(messages: list[dict]) -> list:
    _MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}
    return [_MAP[m["role"]](content=m["content"]) for m in messages]
```

Replace the chat call inside `answer`:

```python
lc_messages = _to_lc_messages(messages)
response    = self._llm.invoke(lc_messages)
answer_text = response.content.strip()
```

- [ ] **Step 2: Update app/agent.py**

Remove `from langchain_openai import ChatOpenAI`, `CHAT_MODEL`, `TEMPERATURE` constants. Add:

```python
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from providers import build_chat_model
```

In `Agent.__init__`, replace the model construction:

```python
model = build_chat_model()
```

- [ ] **Step 3: Update app/tools.py**

Add after existing imports:

```python
import sys
from pathlib import Path as _Path
_PROJECT_ROOT = _Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from providers import build_chat_model as _build_chat_model
from langchain_core.messages import SystemMessage as _SystemMessage, HumanMessage as _HumanMessage
```

Inside `make_tools()`, add near the top:

```python
_synth_llm = _build_chat_model()
```

Replace the body of the `synthesize` tool OpenAI call:

```python
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
```

- [ ] **Step 4: Smoke-test manually**

```bash
cd app
python -c "from rag_pipeline import RagPipeline; print('rag_pipeline OK')"
python -c "from agent import Agent; print('agent OK')"
python -c "from tools import make_tools; print('tools OK')"
```

Expected: three `OK` lines. If Chroma is not built yet, `RagPipeline()` will raise `RuntimeError` about a missing collection -- that is expected.

- [ ] **Step 5: Commit**

```bash
git add app/rag_pipeline.py app/agent.py app/tools.py
git commit -m "feat: wire provider factories into app layer"
```

---

## Task 4: Wire providers into pipeline/

**Files:**
- Modify: `pipeline/build_vectorstore.py`
- Modify: `pipeline/extract_themes.py`
- Modify: `pipeline/deduplicate_themes.py`

**Interfaces:**
- Consumes: `providers.build_chat_model()`, `providers.build_embedder()`

- [ ] **Step 1: Update pipeline/build_vectorstore.py**

Add after existing imports (remove `from openai import OpenAI`, `EMBED_MODEL` constant):

```python
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config as _config
from providers import build_embedder
COLLECTION = _config.CHROMA_COLLECTION    # replaces hardcoded "impact_florida_docs"
```

Replace `embed_and_index` to accept `embedder` instead of `openai_client`:

```python
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

    metadatas = [
        {
            "file_id":        _safe(rec.get("file_id")),
            "file_name":      _safe(rec.get("file_name")),
            "drive_url":      _safe(rec.get("drive_url")),
            "folder_path":    _safe(rec.get("folder_path")),
            "program":        _safe(rec.get("program")),
            "doc_type":       _safe(rec.get("doc_type")),
            "academic_year":  _safe(rec.get("academic_year")),
            "season":         _safe(rec.get("season")),
            "date_precision": _safe(rec.get("date_precision")),
            "district":       _safe(rec.get("district")),
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
```

Fix `main()` -- replace `OpenAI()` init and fix the `do_rebuild` bug:

```python
def main(rebuild: bool = False) -> None:
    parser = argparse.ArgumentParser(description="Build Chroma vector store")
    parser.add_argument("--rebuild", action="store_true",
                        help="Drop and rebuild the collection from scratch")
    args, _ = parser.parse_known_args()
    do_rebuild = rebuild or args.rebuild          # was: `rebuild or do_rebuild` (undefined variable bug)

    embedder = build_embedder()                   # replaces: openai_client = OpenAI()
    # ... rest of main() unchanged, passing embedder where client was passed ...
    added = embed_and_index(embedder, collection, chunks, rec)
```

- [ ] **Step 2: Update pipeline/extract_themes.py**

Remove `from openai import OpenAI`, `MODEL` constant. Add:

```python
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from providers import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
```

Replace `_call_with_retry`:

```python
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
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
```

Update `extract_themes` signature to `extract_themes(llm: BaseChatModel, record: dict)`.

Update `main()`: replace `client = OpenAI()` with `llm = build_chat_model()` and pass `llm` wherever `client` was passed.

- [ ] **Step 3: Update pipeline/deduplicate_themes.py**

Remove `from openai import OpenAI`, `MODEL` constant. Add:

```python
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from providers import build_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.language_models import BaseChatModel
```

Replace `generate_theme_map`:

```python
def generate_theme_map(llm: BaseChatModel, themes: list[str]) -> dict[str, str]:
    resp = llm.invoke([
        SystemMessage(content=_MAP_SYSTEM),
        HumanMessage(content=_map_prompt(themes)),
    ])
    mapping: dict[str, str] = json.loads(resp.content)
    return {t: mapping.get(t, t) for t in themes}
```

Replace `generate_cluster_map`:

```python
def generate_cluster_map(llm: BaseChatModel, canonical: list[str]) -> dict[str, str]:
    resp = llm.invoke([
        SystemMessage(content=_CLUSTER_SYSTEM),
        HumanMessage(content=_cluster_prompt(canonical)),
    ])
    mapping: dict[str, str] = json.loads(resp.content)
    return {t: mapping.get(t, t) for t in canonical}
```

Update `main()`: replace `client = OpenAI()` with `llm = build_chat_model()` and pass `llm` to both generate functions.

- [ ] **Step 4: Smoke-test each pipeline script**

```bash
cd pipeline
python -c "import build_vectorstore; print('build_vectorstore OK')"
python -c "import extract_themes; print('extract_themes OK')"
python -c "import deduplicate_themes; print('deduplicate_themes OK')"
```

Expected: three `OK` lines. No API calls are made on import.

- [ ] **Step 5: Commit**

```bash
git add pipeline/build_vectorstore.py pipeline/extract_themes.py pipeline/deduplicate_themes.py
git commit -m "feat: wire provider factories into pipeline scripts; fix do_rebuild bug"
```

---

## Task 5: Document source protocol + GoogleDriveSource

**Files:**
- Create: `pipeline/sources/__init__.py`
- Create: `pipeline/sources/base.py`
- Create: `pipeline/sources/gdrive.py`
- Create: `tests/test_sources.py`

**Interfaces:**
- Produces:
  - `DocumentRecord` TypedDict (all fields required by ingest.py)
  - `DocumentSource` Protocol: `fetch_documents(dest_dir: Path) -> list[DocumentRecord]`
  - `GoogleDriveSource(drive_id: str, creds_path: Path)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_gdrive_source_implements_protocol():
    """GoogleDriveSource satisfies DocumentSource at the type level."""
    from pipeline.sources.gdrive import GoogleDriveSource
    from pipeline.sources.base import DocumentSource
    import inspect
    assert hasattr(GoogleDriveSource, "fetch_documents")
    sig = inspect.signature(GoogleDriveSource.fetch_documents)
    assert "dest_dir" in sig.parameters
```

- [ ] **Step 2: Run test -- verify it fails**

```
pytest tests/test_sources.py::test_gdrive_source_implements_protocol -v
```

Expected: `ModuleNotFoundError: No module named pipeline.sources`

- [ ] **Step 3: Create pipeline/sources/__init__.py (empty file)**

- [ ] **Step 4: Create pipeline/sources/base.py**

```python
# pipeline/sources/base.py
"""
DocumentSource protocol and DocumentRecord TypedDict.

Both pipeline/sources/gdrive.py and pipeline/sources/local.py must produce
list[DocumentRecord] from their fetch_documents() method. The rest of the
pipeline (ingest.py, build_vectorstore.py) reads these fields unchanged.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, TypedDict


class DocumentRecord(TypedDict):
    file_id:         str
    file_name:       str
    mime_type:       str
    folder_path:     str
    local_path:      str          # absolute path to the downloaded/existing file
    drive_url:       str          # empty string for local source
    program:         str | None
    doc_type:        str | None
    academic_year:   str | None
    season:          str | None
    date_precision:  str          # "direct" | "month_derived" | "unknown"
    district:        str | None
    download_status: str          # "downloaded" | "exists" | "error" | "skipped"


class DocumentSource(Protocol):
    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]:
        """
        Populate dest_dir with files (if needed) and return one DocumentRecord
        per document. dest_dir will be created if it does not exist.
        """
        ...
```

- [ ] **Step 5: Create pipeline/sources/gdrive.py**

```python
# pipeline/sources/gdrive.py
"""
Google Drive document source.

Delegates to the existing get_docs.py functions so we do not duplicate
Drive API logic. Returns the same metadata structure that get_docs.main()
would write to metadata.json, as a list[DocumentRecord].
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from pipeline.sources.base import DocumentRecord

_PIPELINE_DIR = Path(__file__).parent.parent
if str(_PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(_PIPELINE_DIR))


class GoogleDriveSource:
    """Fetch documents from a Google Shared Drive."""

    def __init__(self, drive_id: str, creds_path: Path | None = None) -> None:
        self.drive_id   = drive_id
        self.creds_path = creds_path

    def fetch_documents(self, dest_dir: Path) -> list[DocumentRecord]:
        """
        Download all files from the Shared Drive into dest_dir and return
        their metadata as DocumentRecord dicts.

        Delegates to get_docs.fetch_all_documents() which is extracted from
        get_docs.main() in Step 6 of this task.
        """
        import get_docs

        if self.drive_id:
            os.environ["SHARED_DRIVE_ID"] = self.drive_id

        service = get_docs.build_drive_service()
        records = get_docs.fetch_all_documents(service, self.drive_id, dest_dir)
        return records
```

- [ ] **Step 6: Extract fetch_all_documents from pipeline/get_docs.py**

Open `pipeline/get_docs.py`. The existing `main()` function drives the full download loop. Extract the core loop into a new function `fetch_all_documents` that returns a list of metadata dicts and call it from `main()`:

```python
def fetch_all_documents(service, drive_id: str, dest_dir: Path) -> list[dict]:
    """
    Walk the Shared Drive, download all files into dest_dir, and return
    a list of metadata dicts (same structure as metadata.json records).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    # Move the body of the existing main() download loop here.
    # The return value is the list of record dicts that main() previously
    # wrote directly to metadata.json.
    ...

def main() -> None:
    # ... existing argument parsing unchanged ...
    service = build_drive_service()
    records = fetch_all_documents(service, SHARED_DRIVE_ID, DATA_DIR)
    METADATA_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
```

The exact implementation of the loop body depends on the current `main()` in `get_docs.py` -- read that file and move the download loop as-is.

- [ ] **Step 7: Run test -- verify it passes**

```
pytest tests/test_sources.py::test_gdrive_source_implements_protocol -v
```

Expected: 1 passed

- [ ] **Step 8: Commit**

```bash
git add pipeline/sources/__init__.py pipeline/sources/base.py pipeline/sources/gdrive.py pipeline/get_docs.py tests/test_sources.py
git commit -m "feat: add DocumentSource protocol and GoogleDriveSource"
```

---

## Task 6: LocalFolderSource

**Files:**
- Create: `pipeline/sources/local.py`

**Interfaces:**
- Consumes: `pipeline.sources.base.DocumentRecord`, `pipeline.sources.base.DocumentSource`
- Produces: `LocalFolderSource(root: Path)` satisfying `DocumentSource`

- [ ] **Step 1: Write the failing tests**

```python
# Add to tests/test_sources.py

from pipeline.sources.local import LocalFolderSource


def test_local_source_lists_files(tmp_path):
    (tmp_path / "report.pdf").write_bytes(b"fake pdf")
    (tmp_path / "survey.docx").write_bytes(b"fake docx")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert len(docs) == 2
    names = {d["file_name"] for d in docs}
    assert names == {"report.pdf", "survey.docx"}


def test_local_source_empty_dir(tmp_path):
    source = LocalFolderSource(tmp_path)
    assert source.fetch_documents(tmp_path) == []


def test_local_source_skips_hidden_files(tmp_path):
    (tmp_path / ".DS_Store").write_bytes(b"x")
    (tmp_path / "visible.pdf").write_bytes(b"y")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["file_name"] == "visible.pdf"


def test_local_source_skips_unsupported_extensions(tmp_path):
    (tmp_path / "image.jpg").write_bytes(b"jpg")
    (tmp_path / "archive.zip").write_bytes(b"zip")
    (tmp_path / "doc.pdf").write_bytes(b"pdf")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0]["file_name"] == "doc.pdf"


def test_local_source_infers_program_from_top_folder(tmp_path):
    prog_dir = tmp_path / "SWS"
    prog_dir.mkdir()
    (prog_dir / "notes.pdf").write_bytes(b"x")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert docs[0]["program"] == "SWS"


def test_local_source_download_status_is_exists(tmp_path):
    (tmp_path / "file.pdf").write_bytes(b"x")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert docs[0]["download_status"] == "exists"


def test_local_source_file_id_is_stable(tmp_path):
    (tmp_path / "file.pdf").write_bytes(b"x")
    source = LocalFolderSource(tmp_path)
    docs1 = source.fetch_documents(tmp_path)
    docs2 = source.fetch_documents(tmp_path)
    assert docs1[0]["file_id"] == docs2[0]["file_id"]


def test_local_source_local_path_is_absolute(tmp_path):
    (tmp_path / "file.pdf").write_bytes(b"x")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert Path(docs[0]["local_path"]).is_absolute()
```

- [ ] **Step 2: Run tests -- verify they fail**

```
pytest tests/test_sources.py -k "local" -v
```

Expected: `ModuleNotFoundError: No module named pipeline.sources.local`

- [ ] **Step 3: Create pipeline/sources/local.py**

```python
# pipeline/sources/local.py
"""
Local folder document source.

Walks a directory tree and returns a DocumentRecord for each supported file.
No download is performed -- local_path points to the file directly.
drive_url is always empty.
file_id is a stable SHA-1 hash of the file path relative to root.

Folder structure convention (optional -- users can reorganise as needed):
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
            parts    = rel.parts  # (program, doc_type, ..., filename) or fewer

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
```

- [ ] **Step 4: Run tests -- verify they pass**

```
pytest tests/test_sources.py -k "local" -v
```

Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add pipeline/sources/local.py tests/test_sources.py
git commit -m "feat: add LocalFolderSource for file-system document ingestion"
```

---

## Task 7: Wire source abstraction into pipeline/__init__.py

**Files:**
- Modify: `pipeline/__init__.py`

**Interfaces:**
- Consumes: `config.SOURCE_TYPE`, `config.LOCAL_DOCS_DIR`, `config.SHARED_DRIVE_ID`
- Consumes: `pipeline.sources.gdrive.GoogleDriveSource`, `pipeline.sources.local.LocalFolderSource`

- [ ] **Step 1: Update pipeline/__init__.py**

Add the import block near the top (after existing imports):

```python
import sys
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import config as _config
```

Replace Steps 2-3 in `main()` -- the section that calls `get_docs` and `ingest`:

```python
# ------------------------------------------------------------------
# Step 2: Fetch documents (Drive or local)
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

elif stream and not DOCUMENTS_PATH.exists():
    # Streaming path (gdrive only)
    print("=== Steps 2-3 (stream): downloading and extracting without data/raw/ ===")
    drive_id = _config.SHARED_DRIVE_ID
    if not drive_id:
        raise EnvironmentError("SHARED_DRIVE_ID is not set in .env")

    from get_docs import build_drive_service, stream_docs as _stream_docs
    from ingest import process_stream

    service       = build_drive_service()
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
    print("[skip] Steps 2-3 -- documents.parquet already exists")

else:
    # Standard gdrive download path
    if _has_raw_files():
        file_count = sum(1 for _ in DATA_DIR.iterdir())
        print(f"[skip] Step 2  -- data/raw/ already has {file_count} files")
    else:
        print("=== Step 2: Downloading documents ===")
        from get_docs import main as run_get_docs
        run_get_docs()

    # ------------------------------------------------------------------
    # Step 3: Extract text
    # ------------------------------------------------------------------
    if DOCUMENTS_PATH.exists():
        print("[skip] Step 3  -- documents.parquet already exists")
    else:
        print("=== Step 3: Extracting text ===")
        from ingest import main as run_ingest
        run_ingest()
```

Note: for `SOURCE_TYPE=local`, Step 3 (text extraction via `ingest.py`) runs unchanged after Step 2. The `ingest.py` script reads `local_path` from the metadata records to extract text from those files.

- [ ] **Step 2: Smoke-test**

```bash
python -c "
import os; os.environ['SOURCE_TYPE'] = 'local'; os.environ['LOCAL_DOCS_DIR'] = '.'
import pipeline; print('pipeline __init__ OK')
"
```

Expected: `pipeline __init__ OK` with no import errors.

- [ ] **Step 3: Commit**

```bash
git add pipeline/__init__.py
git commit -m "feat: wire LocalFolderSource into pipeline orchestrator"
```

---

## Task 8: requirements.txt + README note

**Files:**
- Modify: `requirements.txt`
- Modify: `README.md`

- [ ] **Step 1: Update requirements.txt**

Remove the existing `langchain-openai>=0.1.0` line and replace with:

```
# ------------------------------------------------------------------
# LLM providers -- install only the one(s) you intend to use.
# langchain-openai is required (used as the Ollama adapter too).
# ------------------------------------------------------------------
langchain-openai>=0.2.0

# Anthropic (set LLM_PROVIDER=anthropic)
langchain-anthropic>=0.3.0

# Google Gemini (set LLM_PROVIDER=google or EMBED_PROVIDER=google)
langchain-google-genai>=2.0.0

# Ollama / local models (set LLM_PROVIDER=ollama or EMBED_PROVIDER=ollama)
langchain-ollama>=0.2.0
```

- [ ] **Step 2: Update README.md -- add Provider Configuration section**

Insert after the existing "Setup" section:

````markdown
## Provider Configuration

Copy `.env.example` to `secrets/.env` and set the provider you want:

```bash
# OpenAI (default)
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=<your-openai-api-key>

# Anthropic -- embeddings handled automatically via local Ollama
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
ANTHROPIC_API_KEY=<your-anthropic-api-key>
# Ollama must be running: https://ollama.com

# Google Gemini
LLM_PROVIDER=google
LLM_MODEL=gemini-2.0-flash
GOOGLE_API_KEY=<your-google-api-key>

# Fully local via Ollama
LLM_PROVIDER=ollama
LLM_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434  (default)
```

> **Changing embedding models** requires rebuilding the vector store:
> `python pipeline/__init__.py --override`

## Document Source

```bash
# Google Drive (default)
SOURCE_TYPE=gdrive
SHARED_DRIVE_ID=your-drive-id

# Local folder
SOURCE_TYPE=local
LOCAL_DOCS_DIR=/path/to/your/documents
```
````

- [ ] **Step 3: Install new packages**

```bash
pip install langchain-anthropic langchain-google-genai langchain-ollama
```

- [ ] **Step 4: Run full test suite**

```
pytest tests/ -v
```

Expected: all tests pass (13+ tests across test_config, test_providers, test_sources).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt README.md
git commit -m "docs: add provider and source configuration to requirements and README"
```

---

## Self-Review

### Spec Coverage

| Requirement | Task |
|---|---|
| Central config (env vars) | Task 1 |
| OpenAI chat | Task 2 |
| Anthropic chat | Task 2 |
| Google Gemini chat | Task 2 |
| Ollama local chat | Task 2 |
| OpenAI embeddings | Task 2 |
| Google embeddings | Task 2 |
| Ollama local embeddings (auto for Anthropic + Ollama) | Task 1 + 2 |
| Wire chat into app layer | Task 3 |
| Wire embeddings into pipeline | Task 4 |
| Wire chat into pipeline scripts | Task 4 |
| Fix do_rebuild bug in build_vectorstore.py | Task 4 |
| Fix COLLECTION constant in build_vectorstore.py | Task 4 |
| Google Drive source | Task 5 |
| Local folder source | Task 6 |
| Source abstraction in orchestrator | Task 7 |
| requirements.txt + README | Task 8 |

### Known Limitations

- `response_format={"type": "json_object"}` is removed from pipeline LLM calls. The prompts instruct JSON output, which works across all providers, but is less strictly enforced than OpenAI JSON mode. If a model returns malformed JSON, `json.loads()` will raise and the retry loop in `extract_themes.py` will catch it and retry.
- The `GoogleDriveSource` in Task 5 Step 6 requires extracting `fetch_all_documents` from `get_docs.main()`. The exact extraction is marked with an implementation note -- the implementer must read `get_docs.py` to do this correctly.
- Streaming (`--stream`) remains Google Drive only. Running with `SOURCE_TYPE=local --stream` will fall through to the local path (streaming is not relevant for local files).