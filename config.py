"""
Central configuration -- reads from secrets/.env.
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

# Folder depth -> tag name for LocalFolderSource.
# Comma-separated names: depth 0 = first name, depth 1 = second, etc.
# E.g., FOLDER_METADATA_LEVELS=department,category maps:
#   root/HR/Reports/file.pdf  ->  tags = {"department": "HR", "category": "Reports"}
FOLDER_METADATA_LEVELS: list[str] = [
    s.strip()
    for s in os.getenv("FOLDER_METADATA_LEVELS", "").split(",")
    if s.strip()
]

# ------------------------------------------------------------------
# Google Drive  (only needed when SOURCE_TYPE=gdrive)
# ------------------------------------------------------------------
SHARED_DRIVE_ID = os.getenv("SHARED_DRIVE_ID", "")

# ------------------------------------------------------------------
# Vector store
# ------------------------------------------------------------------
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")