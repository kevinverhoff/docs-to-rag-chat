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
