"""Tests for tags-based schema in ingest output."""
import json
import pytest
from pathlib import Path


def _make_metadata(tmp_path, extra_fields=None):
    import json as _json
    record = {
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
    }
    if extra_fields:
        record.update(extra_fields)
    return [record]


def test_ingest_main_writes_tags_json_column(tmp_path, monkeypatch):
    """ingest.main() writes a 'tags' JSON column, not individual program/district columns."""
    import json as _json
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    import importlib
    import ingest

    metadata = _make_metadata(tmp_path)
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(_json.dumps(metadata), encoding="utf-8")
    (tmp_path / "report.pdf").write_bytes(b"%PDF-1.4 fake")

    out_path = tmp_path / "documents.parquet"
    monkeypatch.setattr(ingest, "METADATA_PATH", meta_path)
    monkeypatch.setattr(ingest, "OUTPUT_PATH", out_path)
    ingest.main()

    import pandas as pd
    df = pd.read_parquet(out_path)
    assert "tags" in df.columns, "Expected 'tags' column in output"
    assert "program" not in df.columns, "Expected 'program' column to be removed"
    assert "district" not in df.columns, "Expected 'district' column to be removed"
    tags = _json.loads(df.iloc[0]["tags"])
    assert tags.get("program") == "Ekumen Outreach"
    assert tags.get("district") == "Gethen"
    assert tags.get("academic_year") == "2024-25"


def test_ingest_tags_excludes_system_fields(tmp_path, monkeypatch):
    """tags column does not include file_id, file_name, mime_type, etc."""
    import json as _json
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
    import ingest

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
    (tmp_path / "f.pdf").write_bytes(b"%PDF-1.4 x")

    out_path = tmp_path / "documents.parquet"
    monkeypatch.setattr(ingest, "METADATA_PATH", meta_path)
    monkeypatch.setattr(ingest, "OUTPUT_PATH", out_path)
    ingest.main()

    import pandas as pd
    df = pd.read_parquet(out_path)
    tags = _json.loads(df.iloc[0]["tags"])
    for field in ("file_id", "file_name", "mime_type", "drive_url", "folder_path", "download_status"):
        assert field not in tags, f"System field '{field}' should not be in tags"
    assert tags.get("program") == "Ansible Studies"