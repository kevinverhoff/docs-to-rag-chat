import inspect
from pathlib import Path

import pytest


def test_gdrive_source_implements_protocol():
    """GoogleDriveSource has a fetch_documents method with dest_dir parameter."""
    from pipeline.sources.gdrive import GoogleDriveSource
    assert hasattr(GoogleDriveSource, "fetch_documents")
    sig = inspect.signature(GoogleDriveSource.fetch_documents)
    assert "dest_dir" in sig.parameters


# ---------------------------------------------------------------------------
# LocalFolderSource tests
# ---------------------------------------------------------------------------

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


def test_local_source_tags_is_dict(tmp_path):
    (tmp_path / "file.pdf").write_bytes(b"x")
    source = LocalFolderSource(tmp_path)
    docs = source.fetch_documents(tmp_path)
    assert isinstance(docs[0]["tags"], dict)


def test_local_source_tags_populated_from_folder_levels(tmp_path, monkeypatch):
    """Tags are built from FOLDER_METADATA_LEVELS config."""
    import importlib
    import config
    monkeypatch.setenv("FOLDER_METADATA_LEVELS", "department,category")
    importlib.reload(config)

    dept_dir = tmp_path / "HR"
    cat_dir  = dept_dir / "Reports"
    cat_dir.mkdir(parents=True)
    (cat_dir / "file.pdf").write_bytes(b"x")

    source = LocalFolderSource(tmp_path)
    docs   = source.fetch_documents(tmp_path)

    assert docs[0]["tags"].get("department") == "HR"
    assert docs[0]["tags"].get("category")   == "Reports"

    # restore
    monkeypatch.setenv("FOLDER_METADATA_LEVELS", "")
    importlib.reload(config)


def test_local_source_no_tags_without_levels(tmp_path, monkeypatch):
    """When FOLDER_METADATA_LEVELS is empty, tags is an empty dict."""
    import importlib
    import config
    monkeypatch.setenv("FOLDER_METADATA_LEVELS", "")
    importlib.reload(config)

    prog_dir = tmp_path / "SomeFolder"
    prog_dir.mkdir()
    (prog_dir / "file.pdf").write_bytes(b"x")

    source = LocalFolderSource(tmp_path)
    docs   = source.fetch_documents(tmp_path)
    assert docs[0]["tags"] == {}