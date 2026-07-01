# docs-to-rag-chat

A RAG pipeline and Streamlit chat interface for any document collection. Point it at a folder or a Google Drive, run the pipeline, and get a chat interface that answers questions grounded in your documents — with citations back to the source.

---

## What it does

```
Documents (local folder or Google Drive)
     |
     v
Step 2   Document Fetcher      Download or index files; extract structured metadata from folder paths
Step 3   Text Extractor        Extract text from PDFs, Word docs, spreadsheets, slides, CSV, TXT
Step 3.5 Theme Extractor       LLM-extract themes, key findings, and notable quotes per document
Step 3.6 Theme Deduplicator    Canonicalize + cluster themes across the corpus
Steps 4-6 Chunk → Embed → Index  Split text, embed with your chosen model, store in ChromaDB
     |
     v
Step 7   RAG Pipeline          Retrieve relevant chunks → generate a cited answer
Step 8   Agent + Tools         Multi-tool agent for cross-document comparison and theme analysis
Step 9   Chat Interface        Streamlit UI with sidebar filters and Browse Themes tab
```

Steps 2-6 run once to build the knowledge base. Steps 7-9 are what users interact with.

---

## Supported providers

| `LLM_PROVIDER` | Default model | Package | Default embeddings |
|---|---|---|---|
| `openai` _(default)_ | `gpt-4o-mini` | `langchain-openai` | OpenAI |
| `anthropic` | `claude-3-5-haiku-20241022` | `langchain-anthropic` | Ollama (local) |
| `google` | `gemini-2.0-flash` | `langchain-google-genai` | Google |
| `ollama` | `llama3.2` | `langchain-ollama` | Ollama (local) |

Anthropic has no embedding model, so it automatically falls back to Ollama local embeddings.

---

## Supported document sources

| `SOURCE_TYPE` | Description |
|---|---|
| `gdrive` _(default)_ | Download from a Google Shared Drive via service account |
| `local` | Read from a local folder on disk |

---

## Setup

### 1 — Clone and install

```bash
git clone <repo-url>
cd docs-to-rag-chat
pip install -r requirements.txt
```

### 2 — Create `secrets/.env`

```bash
cp .env.example secrets/.env
```

Open `secrets/.env` and fill in values for your chosen provider and source. The example file documents every option.

**Minimum for OpenAI + local folder:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-api-key>

SOURCE_TYPE=local
LOCAL_DOCS_DIR=/path/to/your/documents
FOLDER_METADATA_LEVELS=department,category
```

**Minimum for OpenAI + Google Drive:**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=<your-openai-api-key>

SOURCE_TYPE=gdrive
SERVICE_ACCOUNT_FILE=secrets/service-account.json
SHARED_DRIVE_ID=<your-drive-id>
```

### 3 — (Google Drive only) Configure Drive access

Create a GCP project and enable the Google Drive API:
- GCP Console → APIs & Services → Library → "Google Drive API" → Enable

Create a service account and download the JSON key:
- APIs & Services → Credentials → Create Credentials → Service account
- Keys → Add Key → Create new key → JSON
- Save as `secrets/service-account.json` (gitignored — never commit this)

Share the Shared Drive with the service account email as a **Viewer**.

### 4 — (Anthropic or Ollama) Set up local embeddings

Anthropic has no embedding model. When `LLM_PROVIDER=anthropic` or `LLM_PROVIDER=ollama`, the pipeline uses Ollama local embeddings automatically.

Install [Ollama](https://ollama.com), then pull the embedding model:

```bash
ollama pull nomic-embed-text
```

---

## Provider Configuration

All provider settings are in `secrets/.env`. Full reference:

```env
# LLM
LLM_PROVIDER=openai           # openai | anthropic | google | ollama
LLM_MODEL=gpt-4o-mini         # model name for your chosen provider
LLM_TEMPERATURE=0.1

# Embeddings (auto-selected based on LLM_PROVIDER, override if needed)
EMBED_PROVIDER=openai         # openai | google | ollama
EMBED_MODEL=text-embedding-3-small

# Ollama (when LLM_PROVIDER or EMBED_PROVIDER is ollama)
OLLAMA_BASE_URL=http://localhost:11434
```

---

## Document Source and Metadata

### Local folder

Set `SOURCE_TYPE=local` and `LOCAL_DOCS_DIR` to the path of your document folder.

Use `FOLDER_METADATA_LEVELS` to map folder depth to filterable metadata tags:

```env
SOURCE_TYPE=local
LOCAL_DOCS_DIR=/path/to/documents
FOLDER_METADATA_LEVELS=department,category
```

With this config, the folder structure maps to tags:

```
documents/
├── HR/                     →  department = "HR"
│   ├── Policies/           →  category   = "Policies"
│   │   └── handbook.pdf
│   └── template.docx       →  category   = null (only one level deep)
└── Finance/                →  department = "Finance"
    └── report.xlsx
```

Tags become filterable dimensions in the sidebar and searchable in the agent.

### Google Drive

Set `SOURCE_TYPE=gdrive`. The pipeline downloads all files from the Shared Drive and parses metadata from folder paths. The folder depth → tag mapping follows the same logic as local source, with the parsed field names stored as tags.

---

## Build the knowledge base

Run the full pipeline from the project root:

```bash
python pipeline/__init__.py
```

Each step is skipped automatically if its output already exists, so it is safe to re-run after an interruption.

**Pipeline flags:**

| Flag | Effect |
|---|---|
| _(none)_ | Resume: skip any step whose output already exists |
| `--override` | Delete all outputs and rerun every step |
| `--override --keep-raw` | Same, but preserve `data/raw/` (skips re-download) |
| `--stream` | Download and extract without writing raw files to `data/raw/` |

**Run steps individually:**

```bash
# Step 3 — extract text
python pipeline/ingest.py

# Step 3.5 — LLM theme extraction
python pipeline/extract_themes.py

# Step 3.6 — canonicalize + cluster themes
python pipeline/deduplicate_themes.py

# Steps 4-6 — chunk, embed, index
python pipeline/build_vectorstore.py
python pipeline/build_vectorstore.py --rebuild   # start fresh
```

---

## Run the app

```bash
streamlit run app/app.py
```

Open `http://localhost:8501`. The sidebar filters are built automatically from whatever tag keys exist in your data — no configuration needed.

**CLI access:**

```bash
# Direct RAG query
python app/rag_pipeline.py "What are the main findings?"
python app/rag_pipeline.py "Summarize the HR policies" --filter department=HR

# Agent query
python app/agent.py "What themes appear in Q4 reports?"
python app/agent.py "Compare findings across departments" --filter category=Reports
```

---

## Configuration reference

All values are read from `secrets/.env`. See `.env.example` for the full annotated template.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` \| `anthropic` \| `google` \| `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Model name for the chosen provider |
| `LLM_TEMPERATURE` | `0.1` | Generation temperature |
| `EMBED_PROVIDER` | _(auto)_ | `openai` \| `google` \| `ollama`; auto-selected from `LLM_PROVIDER` |
| `EMBED_MODEL` | _(auto)_ | Embedding model; auto-selected from `EMBED_PROVIDER` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SOURCE_TYPE` | `gdrive` | `gdrive` \| `local` |
| `LOCAL_DOCS_DIR` | _(empty)_ | Absolute path to local document folder |
| `FOLDER_METADATA_LEVELS` | _(empty)_ | Comma-separated tag names per folder depth |
| `SERVICE_ACCOUNT_FILE` | _(empty)_ | Path to GCP service account JSON (gdrive only) |
| `SHARED_DRIVE_ID` | _(empty)_ | Google Shared Drive ID (gdrive only) |
| `CHROMA_COLLECTION` | `docs` | ChromaDB collection name |

---

## How the agent tools work

The agent has two groups of tools:

**Q&A tools** (vector store retrieval):

| Tool | When to use |
|---|---|
| `search` | Discover which documents exist on a topic |
| `answer` | Specific factual questions — retrieve → generate → cite |
| `summarize` | High-level synthesis across many documents |
| `extract_quotes` | Verbatim passages with source attribution |

**Cross-dimensional tools** (themes layer):

| Tool | When to use |
|---|---|
| `browse_themes` | Open-ended exploration of pre-extracted themes |
| `compare` | Semantic search grouped by a tag dimension for side-by-side synthesis |
| `synthesize` | Write a lead paragraph from the most recent `compare()` call |

All tools accept an optional `filters` dict of tag key-value pairs. Active sidebar filters are injected into every tool call automatically.

The agent's system prompt lives in `prompts/agent_system_prompt.txt`. Edit that file to change how the agent responds — no code changes needed.

---

## File structure

```
docs-to-rag-chat/
├── README.md
├── requirements.txt
├── config.py                              Central env var config
├── providers.py                           LLM + embedding factory functions
├── .env.example                           Template for secrets/.env
├── secrets/                               Gitignored — credentials and .env live here
│
├── pipeline/                              Steps 2-6: build the knowledge base
│   ├── __init__.py                        Orchestrator (--override, --keep-raw, --stream)
│   ├── get_docs.py                        Step 2 (gdrive): download from Drive
│   ├── ingest.py                          Step 3: text extraction → documents.parquet
│   ├── extract_themes.py                  Step 3.5: LLM theme extraction → themes_raw.parquet
│   ├── deduplicate_themes.py              Step 3.6: canonicalize themes → themes.parquet
│   ├── build_vectorstore.py               Steps 4-6: chunk, embed, index → chroma_db/
│   ├── explore_drive.py                   Step 1: Drive structure report (gdrive only)
│   └── sources/
│       ├── base.py                        DocumentRecord TypedDict + DocumentSource Protocol
│       ├── local.py                       LocalFolderSource — walks a local directory
│       └── gdrive.py                      GoogleDriveSource — wraps get_docs.py
│
├── app/                                   Steps 7-9: query and chat interface
│   ├── rag_pipeline.py                    Step 7: retrieve + generate
│   ├── tools.py                           Step 8: agent tool definitions
│   ├── agent.py                           Step 8: LangGraph ReAct agent
│   └── app.py                             Step 9: Streamlit UI
│
├── prompts/                               Edit these to change agent behavior
│   ├── agent_system_prompt.txt
│   └── rag_pipeline_system_prompt.txt
│
├── tests/
│   ├── test_config.py
│   ├── test_providers.py
│   └── test_sources.py
│
└── data/                                  Generated — gitignored
    ├── raw/                               Downloaded source files (gdrive)
    ├── metadata.json                      Per-file metadata manifest
    ├── documents.parquet                  Extracted text corpus
    ├── themes_raw.parquet                 Raw LLM-extracted themes
    ├── theme_map.json                     Raw → canonical theme mapping (human-editable)
    ├── theme_clusters.json                Canonical → cluster mapping (human-editable)
    ├── themes.parquet                     Three-level theme hierarchy
    └── chroma_db/                         Vector store
```

---

## Dependencies

```
# Google Drive API (SOURCE_TYPE=gdrive only)
google-api-python-client>=2.100.0
google-auth>=2.22.0
google-auth-httplib2>=0.2.0

# LLM providers (install only what you use)
langchain>=0.2.0
langchain-openai>=0.2.0       # openai + ollama
langchain-anthropic>=0.3.0    # anthropic
langchain-google-genai>=2.0.0 # google
langchain-ollama>=0.2.0       # ollama embeddings
langgraph>=0.1.0

# Vector store
chromadb>=0.5.0

# Document text extraction
pdfplumber>=0.10.0
python-docx>=1.1.0
python-pptx>=0.6.23
openpyxl>=3.1.0
xlrd>=2.0.0

# Data
pandas>=2.0.0
pyarrow>=14.0.0

# Config / UI
python-dotenv>=1.0.0
streamlit>=1.35.0
```