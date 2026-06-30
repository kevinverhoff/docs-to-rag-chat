import importlib
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
