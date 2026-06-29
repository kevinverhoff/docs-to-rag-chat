# Docs-to-RAG-Chat

<div align="center">
  <!-- Add your logo here: <img src="media/logo.png" alt="Logo" width="480"> -->
</div>

Research assistant for program documents stored in Google Drive.
Staff ask plain-language questions and get cited, grounded answers drawn directly
from the document library. Nothing is made up; every answer traces back to a
specific document.

---

## What We Are Building

```
Google Drive
     |
     v
Step 1   Drive Explorer        Understand the file structure and document inventory
Step 2   Document Fetcher      Download files + parse structured metadata from folder paths
Step 2.5 Doc Type Overrides    Apply manual doc_type corrections -> config/doc_type_overrides.json
Step 3   Text Extractor        Pull readable text from PDFs, Word docs, spreadsheets, slides
Step 3.5 Theme Extractor       LLM-extract themes per document  ->  themes_raw.parquet
Step 3.6 Theme Deduplicator    Canonicalize + cluster themes   ->  theme_map.json + theme_clusters.json + themes.parquet
Step 4   Chunker               Break documents into searchable pieces, preserving structure
Step 5   Embedder              Convert each piece into a vector (text-embedding-3-small)
Step 6   Vector Store          Index all vectors for fast similarity search (Chroma)
     |
     v
Step 7   RAG Pipeline          Answer a question: find relevant chunks -> generate -> cite
Step 8   Agent + Tools         Cross-dimensional themes/trends analysis across programs, districts, time
Step 9   Chat Interface        Staff-facing Streamlit UI
```

Steps 1-6 run once to build the knowledge base. Steps 3.5-3.6 run once after
text extraction to build the themes layer. Steps 7-9 are what staff interact with.

| Step | Component | Status |
|------|-----------|--------|
| 1 | Drive Explorer | ✅ Built |
| 2 | Document Fetcher | ✅ Built |
| 2.5 | Doc Type Override Builder | ✅ Built |
| 3 | Text Extractor | ✅ Built |
| 3.5 | Theme Extractor | ✅ Built |
| 3.6 | Theme Deduplicator | ✅ Built |
| 4 | Chunker | ✅ Built |
| 5 | Embedder | ✅ Built |
| 6 | Vector Store | ✅ Built |
| 7 | RAG Pipeline | ✅ Built |
| 8 | Agent + Tools | ✅ Built |
| 9 | Chat Interface | ✅ Built |

---

## Setup

### Step 1 — Clone and install

```bash
git clone <repo-url>
cd docs-to-rag-chat
pip install -r requirements.txt
```

### Step 2 — Configure Google Drive access

**Create a GCP project** and enable the Google Drive API:
- GCP Console → APIs & Services → Library → "Google Drive API" → Enable

**Create a service account and download the JSON key:**
- APIs & Services → Credentials → Create Credentials → Service account
- Grant the service account no GCP roles (it only needs Drive access)
- In the service account detail page: Keys → Add Key → Create new key → JSON
- Save the downloaded file into the `secrets/` folder, e.g.:
  ```
  secrets/service-account.json
  ```
  This folder is gitignored — never commit credential files.

**Share the Shared Drive with the service account:**
- Open the Shared Drive in Google Drive → Gear icon → Manage members
- Add the service account's `client_email` (from the JSON file) as a **Viewer**

### Step 3 — Create `secrets/.env`

Create a file at `secrets/.env` (gitignored) with the following three variables:

```env
SERVICE_ACCOUNT_FILE=secrets/service-account.json
SHARED_DRIVE_ID=your-drive-id-here
OPENAI_API_KEY=sk-...
```

| Variable | Description |
|---|---|
| `SERVICE_ACCOUNT_FILE` | Path to the GCP service account JSON key, relative to the project root |
| `SHARED_DRIVE_ID` | The alphanumeric ID at the end of the Shared Drive URL: `https://drive.google.com/drive/folders/<ID>` |
| `OPENAI_API_KEY` | Your OpenAI API key — used for embeddings, LLM theme extraction, and the chat agent |

### Step 4 — Build the knowledge base

Run the full pipeline in one command. **Expect this to take 20–60 minutes** on a
full corpus: downloading from Google Drive, running LLM calls for theme extraction
(~$0.50 at gpt-4o-mini pricing), and building the ChromaDB vector store all run
sequentially. Each step is skipped automatically if its output already exists, so
it is safe to re-run after an interruption.

```bash
python pipeline/__init__.py
```

> **First run checklist:** confirm `secrets/.env` is populated and
> `secrets/service-account.json` exists before running. The pipeline will fail
> fast at Step 2 if the Drive connection cannot be established.

**Pipeline flags**

| Flag | Effect |
|---|---|
| _(none)_ | Resume: skip any step whose output already exists |
| `--override` | Delete all outputs and rerun every step from scratch |
| `--override --keep-raw` | Same, but preserve `data/raw/` to skip re-downloading |
| `--stream` | Download and extract without saving files to `data/raw/` |

```bash
# Full rerun from scratch (re-downloads everything from Google Drive)
python pipeline/__init__.py --override

# Full rerun, but reuse files already in data/raw/
python pipeline/__init__.py --override --keep-raw

# First-time run that never writes raw files to disk
python pipeline/__init__.py --stream
```

**When to use each:**
- `--override` — added new documents to the Drive and want a clean rebuild
- `--override --keep-raw` — need to rebuild themes or the vector store but not re-download (saves time and API quota)
- `--stream` — disk space is a constraint, or you prefer not to stage raw files locally

Or run steps individually if you want more control:

```bash
# Step 2 — download all documents from Google Drive
python pipeline/get_docs.py
# Output: data/raw/{file_id}.{ext}  +  metadata.json

# Step 3 — extract text from every file
python pipeline/ingest.py
# Output: documents.parquet

# Step 3.5 — LLM theme extraction (one gpt-4o-mini call per document)
# Cost: ~333 docs x ~750 tokens ≈ $0.50
python pipeline/extract_themes.py
# Output: themes_raw.parquet

# Step 3.6 — canonicalise + cluster themes (two LLM calls total)
python pipeline/deduplicate_themes.py
# Output: theme_map.json  +  theme_clusters.json  +  themes.parquet

# Steps 4-6 — chunk, embed, and index into Chroma
# Cost: ~333 docs x ~20 chunks x ~200 tokens ≈ $0.03
python pipeline/build_vectorstore.py
# Output: chroma_db/
```

### Step 5 — Launch the app

Once `chroma_db/` exists, start the Streamlit interface:

```bash
streamlit run app/app.py
```

Open the URL printed in the terminal (default: `http://localhost:8501`). The sidebar
filters let you scope questions by program, district, year, doc type, or theme cluster.

You can also query the pipeline directly from the CLI:

```bash
python app/rag_pipeline.py "What challenges have districts reported with SWS?"
python app/rag_pipeline.py "What are teachers saying?" --district Lake --program SWS
```

> **Note on Step 3.5:** if the run is interrupted (e.g. API quota exceeded),
> re-run the same command — already-processed documents are skipped automatically.
> To rebuild the vector store after adding credits, run
> `python pipeline/build_vectorstore.py` (Steps 4-6 also resume from where they
> left off, or use `--rebuild` to start fresh).

---

## Pipeline Design

### Guiding Principles

- **Every answer cites its source.** The system retrieves real document passages
  and grounds the response in them. It does not draw on general knowledge.
- **Folder structure is metadata.** Program area, document type, district, and
  academic year are parsed from Drive folder paths — no manual tagging required.
- **Themes are pre-computed.** Per-document theme extraction at ingest time means
  the browse interface is fast and the agent has structured context to work from.

---

### Step 1: Drive Explorer (`pipeline/explore_drive.py`) — ✅ Built

Walks the entire Shared Drive and produces a report: folder hierarchy with file
counts, file type breakdown, totals. Output written to `drive_structure.txt`.
No files downloaded.

---

### Step 2: Document Fetcher (`pipeline/get_docs.py`) — ✅ Built

Downloads every file and parses structured metadata from its folder path.

**Google-native formats are exported:**
`Docs → .docx`, `Sheets → .xlsx`, `Slides → .pptx`, `Forms → .pdf`

Files are saved as `data/raw/{file_id}.{ext}`. Idempotent — already-downloaded
files are skipped.

**Metadata fields written to `metadata.json`:**

| Field | Description |
|---|---|
| `file_id` | Google Drive file ID |
| `file_name` | Original filename |
| `mime_type` | File format |
| `folder_path` | Full path from drive root, slash-separated |
| `local_path` | Path to downloaded file (relative to project root) |
| `drive_url` | Direct Google Drive link (for citations) |
| `program` | Parsed from path: SWS, Focus K-3, Math Materials, Teacher Workforce, EIR/GBL, SPARK, NAT HQIM, Multiple Programs |
| `doc_type` | Parsed from path + filename rules, then overridden by `doc_type_overrides.json`. 16 types: `feedback_survey_data`, `teacher_practice_data`, `qualitative_theming`, `progress_summary`, `engagement_notes`, `impact_data_report`, `evaluation_report`, `grants_and_funder_reporting`, `intake_survey`, `program_content`, `program_overview`, `program_logistics`, `district_artifact`, `field_influence`, `other_data_file`. See [docs/doc_types.md](docs/doc_types.md). |
| `academic_year` | Parsed from path using July boundary rule (e.g. `"2025-26"`) |
| `season` | Fall (Jul-Dec) or Spring (Jan-Jun), null if no month in path |
| `date_precision` | `direct` / `month_derived` / `unknown` |
| `district` | Lake, Lee, Osceola, Pasco, Polk, St. Lucie, etc. — null for statewide docs |
| `download_status` | `downloaded` / `exists` / `error` |
| `downloaded_at` | UTC timestamp |

---

### Step 2.5: Doc Type Override Builder (`pipeline/build_overrides.py`) — ✅ Built

Reads `data/metadata.json` and applies manual corrections to the `doc_type` field.
Writes `config/doc_type_overrides.json`, which is loaded by the ingestion and
vector-store steps to override any auto-parsed doc type.

Run standalone to rebuild overrides after editing the `EXCEPTIONS` list inside
the script, or it runs automatically as part of the full pipeline orchestrator.

---

### Step 3: Text Extractor (`pipeline/ingest.py`) — ✅ Built

Extracts readable text from every downloaded file. Dispatches on file extension.

| Format | Tool | Heading detection |
|---|---|---|
| PDF | `pdfplumber` | No |
| DOCX / DOC | `python-docx` | Yes — H1/H2/H3 with char offsets |
| XLSX / XLS | `openpyxl` | Sheet names as H1 |
| PPTX | `python-pptx` | Title placeholders as H1 |
| CSV | `pandas` | Column headers as H1 |
| TXT | built-in | No |

Heading offsets feed directly into header-aware chunking in Step 4: chunks
do not cross section boundaries.

**Output:** `documents.parquet` — one row per document. Columns: all metadata
fields from `metadata.json` plus `text`, `headings` (JSON), `char_count`,
`extraction_status`, `extraction_error`.

---

### Step 3.5: Theme Extractor (`pipeline/extract_themes.py`) — ✅ Built

One LLM call per document (`gpt-4o-mini`). For each document, extracts:

```json
{
  "themes": ["teacher buy-in", "data use in instruction", "time constraints"],
  "key_findings": ["Districts reported...", "Teachers consistently noted..."],
  "notable_quotes": ["Verbatim sentence pulled directly from the document text."],
  "inferred_academic_year": "2024-25",
  "inferred_season": "Fall"
}
```

**`notable_quotes` must be exact, verbatim text copied from the document** — no
paraphrasing, no summarizing. The LLM prompt explicitly instructs this.

`inferred_academic_year` and `inferred_season` are only populated when
`date_precision == "unknown"` from the folder path.

**Output:** `themes_raw.parquet` — one row per document, all metadata fields
carried through plus the extracted fields above.

**Cost:** ~327 calls x ~750 tokens = approx. $0.50 at gpt-4o-mini pricing.

---

### Step 3.6: Theme Deduplicator (`pipeline/deduplicate_themes.py`) — ✅ Built

Builds a three-level theme hierarchy across two LLM passes and writes two
human-editable mapping files plus the final `themes.parquet`.

**Three levels in `themes.parquet`:**

| Column | Content | Source |
|---|---|---|
| `themes_raw` | Original extracted labels, never altered | Carried from `themes_raw.parquet` |
| `themes` | Canonical labels after dedup | `theme_map.json` applied |
| `theme_clusters` | Unique superordinate clusters for this document | `theme_clusters.json` applied |

**ONLY the theme hierarchy columns are derived from LLM calls.**
`key_findings` and `notable_quotes` are copied through unchanged.

**Pass 1 — raw → canonical (`theme_map.json`):**
Collects all unique raw theme strings from `themes_raw.parquet` and sends them
in a single LLM call to produce a flat mapping of synonyms and near-synonyms
to clean canonical labels (2-5 words each).

**Pass 2 — canonical → cluster (`theme_clusters.json`):**
Takes the full set of canonical labels and assigns each to a broad superordinate
cluster. Produces 5-8 clusters spanning the corpus (e.g., "Program Implementation",
"Data Use & Evidence", "Educator Development", "Student Learning").

Both mapping files are independently skippable — if the file already exists,
that LLM call is skipped. Delete a file to regenerate it without affecting the
other. Edit either file manually and re-run to apply changes.

**Output:** `theme_map.json` + `theme_clusters.json` + `themes.parquet`

The run summary prints document count per cluster — useful for spotting
over-merging or gaps before moving to the browse interface.

---

### Steps 4-6: Chunker + Embedder + Vector Store (`pipeline/build_vectorstore.py`) — ✅ Built

Breaks each document into pieces small enough for semantic search while
keeping enough context for useful answers.

**Header-aware splitting:** DOCX heading boundaries are hard split points.
Chunks never cross section headers. PDFs fall back to paragraph → sentence →
word boundaries.

**Contextual prefix** prepended before embedding (improves retrieval, not shown in answers):

```
File: Site-Visit-Report-Lake-Nov2025.docx
Folder: Data Sources/3_Teacher Workforce/2_District Educator Listening Session Data
Program: Teacher Workforce | District: Lake | Year: 2025-26 | Season: Fall
Section: Key Themes
```

**Chunk parameters:** target 800 chars, 100 char overlap, 100 char minimum.

**Metadata on every chunk:** all six structured fields (`program`, `doc_type`,
`academic_year`, `season`, `date_precision`, `district`) plus `file_id`,
`file_name`, `folder_path`, `section_h1/h2/h3`, `chunk_index`, `chunk_count`.

---

### Step 7: RAG Pipeline (`rag_pipeline.py`) — ✅ Built

Implemented as a `RagPipeline` class. Streamlit initializes it once with
`@st.cache_resource` so the Chroma client and OpenAI client stay open across
re-renders. Chat history lives in `st.session_state`. Each query calls
`pipeline.answer(question, filters)` on the cached instance — no reconnection
per message.

**Query flow:**

```
question
  → embed (text-embedding-3-small)
  → search Chroma, top 8 results, max 2 chunks per doc
  → if question is about themes/trends → join themes.parquet on file_id
  → build prompt (chunks + optional theme context + chat history)
  → generate with gpt-4o-mini (temp 0.1)
  → return {answer, sources}
```

**Theme injection is conditional.** The pipeline detects whether the question
is about patterns, trends, or themes (keyword + intent check) and only pulls
`themes.parquet` rows when relevant. This keeps single-document Q&A fast and
cheap while still supporting cross-document analysis.

**Citations** appear as inline footnotes `[1]`, `[2]` in the answer text, with
a Sources block at the end. Every source always includes the Google Drive link
(`drive_url`) so staff can open the original document in one click.

**Metadata filters** narrow the Chroma search before retrieval. All are
optional kwargs on `pipeline.answer()` and map directly to Streamlit sidebar
dropdowns:

| Filter | Chroma field | Example values |
|---|---|---|
| `program` | `program` | SWS, Focus K-3, Teacher Workforce, … |
| `district` | `district` | Lake, Osceola, Polk, … |
| `academic_year` | `academic_year` | 2024-25, 2025-26 |
| `doc_type` | `doc_type` | feedback_survey_data, teacher_practice_data, qualitative_theming, impact_data_report, … (see [docs/doc_types.md](docs/doc_types.md)) |
| `theme_cluster` | `theme_clusters` | Program Implementation, Educator Development, … |

`theme_cluster` uses Chroma's `$contains` filter against the pipe-separated
`theme_clusters` metadata field set during indexing.

**CLI usage** (run from project root):

```bash
# Simple question
python app/rag_pipeline.py "What challenges have districts reported with SWS?"

# With filters
python app/rag_pipeline.py "What are teachers saying?" --district Lake --program SWS

# Filter by year and theme cluster
python app/rag_pipeline.py "How has teacher buy-in changed?" --year 2024-25 --theme-cluster "Educator Development"

# All options
python app/rag_pipeline.py --help
```

Available filter flags: `--program`, `--district`, `--year`, `--doc-type`, `--theme-cluster`

---

### Step 8: Agent + Tools (`agent.py`, `tools.py`) — ✅ Built

LangGraph ReAct agent (`gpt-4o-mini`) with nine tools across three groups.

**Q&A tools** (vector store):

| Tool | When the agent uses it |
|---|---|
| `search` | Discover which documents exist on a topic — returns titles, programs, districts, similarity scores |
| `answer` | Specific factual questions — full RAG: retrieve → generate → cite with inline links |
| `summarize` | High-level synthesis — broader retrieval across more documents |
| `extract_quotes` | User asks for literal text, pull quotes, or verbatim language from documents |

**Cross-dimensional tools** (`themes.parquet`):

| Tool | When the agent uses it |
|---|---|
| `browse_themes` | Open-ended exploration — "what themes appear in Lake district?" |
| `compare` | Semantic search grouped by `dimension` (`"program"`, `"district"`, or `"academic_year"`) for side-by-side synthesis across the dimension |
| `synthesize` | Write a 2-4 sentence lead paragraph from the most recent `compare()` call — always call immediately after `compare()` |

**Survey statistics tools** (`pipeline/survey_stats.py` + `metadata.json`):

| Tool | When the agent uses it |
|---|---|
| `list_surveys` | Discover which survey spreadsheets are available — call before `survey_stats()` |
| `survey_stats` | Quantitative questions about survey responses — Likert distributions, NPS scores, open-ended samples. Use for: "how did participants rate the convening?", "what % agreed?", "what were the NPS scores?". Column matching is fuzzy: "discussed data" will find "Discuss Data" columns via word-prefix and difflib fallback. |

Survey stats are computed at query time directly from the downloaded `.xlsx`/`.csv`
files — no LLM involved in the math. Column classification is automatic:
- **Likert / categorical** (≤ 10 unique values): response counts and percentages
- **Numeric** (e.g. NPS 0–10): mean, median, min, max
- **Open-ended** (high cardinality): sample verbatim responses
- **Admin columns** (Respondent ID, timestamps): automatically skipped

SurveyMonkey sub-header rows (the "Response / Open-Ended Response" row that
exports as row 2) are detected and stripped automatically.

All tools accept optional filter parameters. Active sidebar filters are injected
into the user message so the agent passes them through to tools automatically.

**Citation design.** The `answer()` and `summarize()` tools wrap their Sources block
in `=== SOURCES ===` markers. The system prompt instructs the agent to copy that block
verbatim — preserving `[n]` numbers and Drive links — rather than reformatting it.
This prevents the agent from renumbering, deduplicating, or dropping sources when it
synthesizes its final response.

#### Python usage

```python
from rag_pipeline import RagPipeline
from agent import Agent

pipeline = RagPipeline()
ag = Agent(pipeline)

result = ag.chat("What coaching challenges have districts reported?")
print(result["answer"])

# With filters
result = ag.chat(
    "How has teacher buy-in changed over time?",
    program="SWS",
    district="Lake",
)
```

#### CLI usage

```bash
python app/agent.py "What challenges have districts reported?"
python app/agent.py "How has teacher buy-in changed?" --program SWS
python app/agent.py "What themes appear in Lake district?" --district Lake
python app/agent.py "Compare Focus K-3 implementation" --doc-type site_visit
python app/agent.py "What clusters under Educator Development?" --theme-cluster "Educator Development"
```

#### Streamlit integration

```python
@st.cache_resource
def load_agent():
    return Agent(RagPipeline())

ag = load_agent()

result = ag.chat(
    prompt,
    history=st.session_state.get("history", []),
    program=st.session_state.get("program_filter"),
    district=st.session_state.get("district_filter"),
)
```

---

### Step 9: Chat Interface (`app.py`) — ✅ Built
```bash
streamlit run app/app.py
```

**Sidebar filters** populate from `themes.parquet` distinct values and scope every agent call.
Choosing "(All)" removes that filter. Available: Program, District, Academic Year, Doc Type,
Theme Cluster.

**Chat tab** — natural language Q&A via the Step 8 agent. Active filters are injected into
each call so all tools (search, answer, compare) respect the current scope. Conversation
history is kept in `st.session_state` and passed to the agent on each turn. "Clear chat"
resets the session.

A **filter-change warning** banner appears just above the chat input when the sidebar
filters change mid-conversation, prompting the user to clear the chat or continue
with the understanding that the context has shifted.

The **Agent Trace** expander (shown after each response) lists every tool call with its
arguments and truncated output. The expander header also shows total **input / output
token counts** for the turn, summed across all LLM calls.

**Browse Themes tab** — filter-aware exploration of `themes.parquet`. Choose a "Group by"
dimension (program / academic year / district / doc type). Each group shows top themes, then
per-document: cluster labels, themes, and the top key finding. Document titles link to Drive.

---

---

## Agent Behavior

The agent's behavior is defined in `prompts/agent_system_prompt.txt`. Edit that file to change how the agent answers — no Python changes needed, just restart the app.

`
You are a research assistant for Impact Florida, an education nonprofit that runs
evidence-based programs in Florida K-12 school districts. You help staff explore
program documents, find evidence, and identify trends across programs and districts.

CRITICAL RULES
1. Every factual claim must come directly from a tool result. Never answer from memory.
2. Never invent, guess, or paraphrase content that was not in a tool result.
3. If tools return no relevant results, say: "I could not find information about that
   in the provided documents." Do not fill in from general knowledge.
4. Never fabricate citations. Only use the ([filename](url)) links returned verbatim
   by a tool. Never write a filename or URL you invented.
5. Never wrap text from answer() or summarize() in quotation marks — those tools
   return synthesized summaries, not verbatim text. Presenting a summary as a quote
   is fabrication. For literal pull quotes, always call extract_quotes().
6. Each quote or factual claim must carry the citation from the exact tool result it
   came from. Never apply a citation from one tool call to content that came from a
   different tool call. If a response combines results from multiple tool calls, every
   distinct source must be cited where its content appears.

COVERAGE — multi-program by default
- Unless the user has explicitly filtered to a single program, every response must
  draw from at least two programs when the corpus supports it.
- If your first tool call returns results from only one program and the question
  is clearly multi-program in scope, make one follow-up call or use compare(topic, dimension="program").
  For narrow questions (specific doc, single district, single year), one program is fine.
- Always name the program explicitly when citing a finding. Never present a finding
  as network-wide without confirming it appears across multiple programs.
- When findings span multiple programs, structure your response program-by-program.
  Use a bold label (e.g., **Reading Rising — Duval County**) before each block.
  Do not merge findings from different programs into a single paragraph.

DEPTH — go one layer deeper
- Do not stop at a high-level finding. For each key claim, include:
    (a) what the finding is
    (b) which program, district, and year it comes from
    (c) at least one supporting detail — an implementation mechanism, outcome
        metric, quoted condition, or contextual factor from the same tool result
- If answer() returns a passage, extract and present its supporting details before
  moving to the next finding. Do not compress specifics into a summary sentence.
- Aim for 2–4 substantive paragraphs per distinct sub-question.
- For key findings, call extract_quotes() to surface a verbatim passage that grounds
  the claim. Prioritize this when the user asks for evidence, examples, or quotes, or
  when a finding would be hard to evaluate without seeing the actual language.

RESPONSE FORMAT — comparison questions
When your response draws on compare() results, always use this structure:

  1. synthesize() output (FIRST — this is the lead)
     2–4 sentences that directly answer the question: what is similar across groups,
     what is different, and what the key takeaway is. Do not save this for the end.
     Example: "Across programs, professional learning is most effective when grounded
     in teachers' own classrooms — but programs differ sharply in how they structure
     that support..."

  2. Per-group detail sections
     One labeled section per program / district / year. Use a bold header
     (e.g., **Focus K-3 — Hillsborough** or **SWS — 2023-24**) before each block.
     Each section: 1–2 sentences of context + specific evidence + inline citation.

  3. Optional closing sentence
     Only if there is a clear implication, gap, or caveat that applies across all groups.

Do NOT open with a per-group section. The synthesis is always the first thing the user reads.

TOOL SELECTION
  Q&A tools (text documents — reports, site visits, memos, presentations):
  - answer()         -- specific factual questions; returns cited passages
  - summarize()      -- high-level overview across many documents
  - compare()        -- semantic search grouped by dimension for side-by-side synthesis;
                        dimension must be one of: "program", "district", "academic_year"
                        USE for: "how does X differ across programs / districts / over time?"
                        Fixed filters (program, district, doc_type) narrow the corpus first.
  - synthesize()     -- write a 2-4 sentence lead paragraph from the most recent compare()
                        call; ALWAYS call this immediately after compare(); reads the compare()
                        result directly -- no need to repeat the passages as an argument
  - browse_themes()  -- explore pre-extracted themes by any filter dimension
  - extract_quotes() -- verbatim passages with source attribution
                        USE for: "pull quotes", "show me what it says", "give me an example",
                        "quotes from", or any request for literal text from a document
  - search()         -- discover which documents exist on a topic

  Survey / spreadsheet tools (quantitative data — Likert ratings, NPS, response counts):
  - list_surveys()      -- discover available survey spreadsheets in the corpus;
                           ALWAYS call this first before survey_stats()
  - survey_stats()      -- descriptive statistics for a survey file:
                           response distributions, means, percentages, sample open-ended responses
                           USE for: "how did teachers rate", "what percentage agreed",
                           "survey results", "satisfaction scores", "NPS", "Likert",
                           or any question about survey responses or participant feedback ratings

  IMPORTANT: survey data lives in spreadsheets, not in the vector store. The Q&A tools
  (answer, summarize, search, etc.) cannot access it. Always use survey_stats() for
  questions about survey response data.

TOOL CHAINING — recommended patterns
  - Pattern or trend questions ("what works", "common challenges", "what do
    programs have in common"):
      1. answer() or summarize() to get initial cited passages
      2. compare(topic, dimension="program") to verify whether the pattern holds
         across programs or is specific to one
      3. extract_quotes() for a grounding example if the pattern needs illustration

  - Specific finding questions ("what did X program do about Y"):
      1. answer() for the finding
      2. extract_quotes() if the user needs to see the actual language

  - Comparison / cross-cutting questions ("how does X differ across programs / districts",
    "how has X evolved", "compare X over time"):
      1. compare(topic, dimension) — retrieve and group semantic search results
      2. synthesize(topic, dimension) — immediately after; writes the lead paragraph
      Present synthesize() output first, then the per-group details from compare().

  - Broad exploratory questions ("tell me about X"):
      1. summarize() for the landscape
      2. compare(topic, dimension="program") + synthesize() for program-by-program view
      3. extract_quotes() for representative examples

  - Survey / quantitative questions ("how did staff rate", "survey results for X",
    "what percentage said Y", "satisfaction with Z"):
      1. list_surveys() to discover available files — pass any active program/district filter
      2. survey_stats() with the exact file name and a question_fragment if the question
         is narrow (e.g., question_fragment="recommend" for NPS-style questions)
      3. If the question also asks for context or quotes from narrative documents,
         follow up with answer() or extract_quotes() after the survey data

  Pass any active sidebar filters (program, district, academic_year, doc_type,
  theme_cluster) through to every tool call.

CITATIONS — required in every response
  - Tool outputs contain inline citations as markdown links, e.g.
    "Teacher retention improved ([Lake_2024_SiteVisit.pdf](https://drive.google.com/...))."
  - Cite immediately at the point of use. Place the ([filename](url)) link directly
    after the sentence or quote it supports — not at the end of a paragraph.
  - Every distinct source used in a response must be cited. If you call three tools
    and each returns content you use, all three sources must appear in the response.
  - Match citations to content. The citation following a quote or claim must come from
    the same tool result that contained that quote or claim. Never borrow a citation
    from a survey tool result and attach it to a quote from a RAG tool result, or
    vice versa.
  - Preserve citations exactly as they appear — never rewrite, shorten, or remove them.
  - Do NOT add a separate SOURCES or References block; all sources are already inline.
  - Never fabricate a citation. Only use links that appear verbatim in a tool result.
`

---

## File Structure

```
team-impact-fl/
├── README.md
├── requirements.txt
├── .env                               (gitignored)
├── config/
│   └── doc_type_overrides.json        Manual doc type corrections (auto-generated, human-editable)
│
├── media/
│   ├── Favicon.png
│   └── LTJ Fellowship Logo Work.png
│
├── pipeline/                          Steps 1-6: build the knowledge base
│   ├── __init__.py                    Orchestrator (--override, --keep-raw, --stream)
│   ├── get_docs.py                    Step 2:   Download from Drive -> data/metadata.json
│   ├── ingest.py                      Step 3:   Text extraction -> data/documents.parquet
│   ├── extract_themes.py              Step 3.5: LLM theme extraction -> data/themes_raw.parquet
│   ├── deduplicate_themes.py          Step 3.6: Canonicalize themes -> data/themes.parquet
│   ├── build_vectorstore.py           Steps 4-6: Chunk, embed, index -> data/chroma_db/
│   ├── explore_drive.py               Step 1:   Drive structure report -> data/drive_structure.txt
│   ├── fetch_guide.py                 One-off:  appends program guide to drive_structure.txt
│   └── build_overrides.py             Utility:  build doc_type_overrides.json
│
├── app/                               Steps 7-9: query and interface
│   ├── rag_pipeline.py                Step 7: Retrieve + generate
│   ├── tools.py                       Step 8: LangGraph tool definitions
│   ├── agent.py                       Step 8: ReAct agent wiring
│   ├── app.py                         Step 9: Streamlit UI  (streamlit run app/app.py)
│   └── feedback.py                    Streamlit feedback component
│
├── prompts/                           System prompts -- edit to change agent behavior
│   ├── agent_system_prompt.txt        Agent rules, tool selection, citation policy
│   └── rag_pipeline_system_prompt.txt RAG answer formatting rules
│
├── docs/
│   └── doc_type_audit.md
│
├── contributors/                      Team notebooks
│
├── deliverables/                      Hackweek outputs
│
└── data/                              Generated -- gitignored
    ├── raw/                           Downloaded source documents
    ├── metadata.json                  Per-file metadata manifest
    ├── documents.parquet              Extracted text corpus
    ├── themes_raw.parquet             Raw LLM-extracted themes
    ├── theme_map.json                 Raw -> canonical label mapping (human-editable)
    ├── theme_clusters.json            Canonical -> cluster mapping (human-editable)
    ├── themes.parquet                 Three-level theme hierarchy
    ├── drive_structure.txt            Drive folder/file inventory
    └── chroma_db/                     Vector store
```

## Dependencies

```
# Google Drive API
google-api-python-client>=2.100.0
google-auth>=2.22.0
google-auth-httplib2>=0.2.0

# OpenAI (embeddings + chat)
openai>=1.30.0

# LangChain / LangGraph (agent)
langchain>=0.2.0
langchain-openai>=0.1.0
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

# Feedback / Google Sheets
gspread>=6.0.0
Pillow>=10.0.0
```



