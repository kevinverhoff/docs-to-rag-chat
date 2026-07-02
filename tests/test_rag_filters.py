"""Tests for generic tag_filters interface in _build_where and answer()."""


def test_build_where_returns_none_for_empty_filters():
    import sys
    sys.path.insert(0, "app")
    from rag_pipeline import _build_where
    assert _build_where(None, None) is None


def test_build_where_single_tag():
    from rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach"}, None)
    assert result == {"program": {"$eq": "Ekumen Outreach"}}


def test_build_where_multiple_tags():
    from rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach", "district": "Gethen"}, None)
    assert result == {
        "$and": [
            {"program":  {"$eq": "Ekumen Outreach"}},
            {"district": {"$eq": "Gethen"}},
        ]
    }


def test_build_where_ignores_none_values():
    from rag_pipeline import _build_where
    result = _build_where({"program": "Ekumen Outreach", "district": None}, None)
    assert result == {"program": {"$eq": "Ekumen Outreach"}}


def test_build_where_theme_cluster():
    from rag_pipeline import _build_where
    result = _build_where(None, "Institutional Change")
    assert result == {"theme_clusters": {"$contains": "Institutional Change"}}


def test_build_where_tag_filters_and_theme_cluster():
    from rag_pipeline import _build_where
    result = _build_where({"program": "Ansible Studies"}, "Equity")
    assert result == {
        "$and": [
            {"program": {"$eq": "Ansible Studies"}},
            {"theme_clusters": {"$contains": "Equity"}},
        ]
    }