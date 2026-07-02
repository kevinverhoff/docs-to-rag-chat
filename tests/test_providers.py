import importlib
import pytest


def _reload_providers(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import config
    importlib.reload(config)
    import providers
    importlib.reload(providers)
    return providers


def test_build_chat_model_openai(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai", LLM_MODEL="gpt-4o-mini",
                          OPENAI_API_KEY="test-key")
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
    p = _reload_providers(monkeypatch, LLM_PROVIDER="google", LLM_MODEL="gemini-2.0-flash",
                          GOOGLE_API_KEY="test-key")
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
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai", OPENAI_API_KEY="test-key")
    from langchain_openai import OpenAIEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, OpenAIEmbeddings)


def test_build_embedder_ollama(monkeypatch):
    # Anthropic auto-selects ollama embeddings
    p = _reload_providers(monkeypatch, LLM_PROVIDER="anthropic")
    from langchain_ollama import OllamaEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, OllamaEmbeddings)


def test_build_embedder_google(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="google", GOOGLE_API_KEY="test-key")
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    emb = p.build_embedder()
    assert isinstance(emb, GoogleGenerativeAIEmbeddings)


def test_build_embedder_unknown_raises(monkeypatch):
    p = _reload_providers(monkeypatch, LLM_PROVIDER="openai", EMBED_PROVIDER="unknown",
                          OPENAI_API_KEY="test-key")
    with pytest.raises(ValueError, match="Unknown EMBED_PROVIDER"):
        p.build_embedder()
