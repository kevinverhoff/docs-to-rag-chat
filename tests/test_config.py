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


def test_load_tags_config_missing_file_gives_empty_dict(tmp_path):
    """_load_tags_config returns {} when tags_config.yaml does not exist."""
    import config
    result = config._load_tags_config(tmp_path / "nonexistent.yaml")
    assert result == {}


def test_load_tags_config_reads_all_sections(tmp_path):
    """_load_tags_config reads program_map, known_districts, district_display, filter_keys."""
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
    result = config._load_tags_config(cfg_path)
    assert result.get("program_map") == {"0_demo": "Demo Program"}
    assert result.get("known_districts") == ["alpha", "beta"]
    assert result.get("district_display") == {"alpha": "Alpha Region"}
    assert result.get("filter_keys") == ["program", "district"]


def test_config_exposes_tags_constants():
    """config module exposes GDRIVE_PROGRAM_MAP, GDRIVE_KNOWN_DISTRICTS, etc. as correct types."""
    import config
    assert isinstance(config.GDRIVE_PROGRAM_MAP, dict)
    assert isinstance(config.GDRIVE_KNOWN_DISTRICTS, list)
    assert isinstance(config.GDRIVE_DISTRICT_DISPLAY, dict)
    assert isinstance(config.FILTER_TAG_KEYS, list)
    # tags_config.yaml is present, so these should be populated
    assert len(config.GDRIVE_PROGRAM_MAP) > 0
    assert len(config.GDRIVE_KNOWN_DISTRICTS) > 0
    assert len(config.FILTER_TAG_KEYS) > 0

def test_get_docs_reads_program_map_from_config():
    """get_docs.PROGRAM_MAP is populated from config (tags_config.yaml), not hardcoded."""
    import pipeline.get_docs as gd
    import config
    # PROGRAM_MAP should be the same object as config.GDRIVE_PROGRAM_MAP
    assert gd.PROGRAM_MAP is config.GDRIVE_PROGRAM_MAP
    # Should contain the Ekumen Outreach entry from tags_config.yaml
    assert "0_ekumen outreach" in gd.PROGRAM_MAP
    assert gd.PROGRAM_MAP["0_ekumen outreach"] == "Ekumen Outreach"