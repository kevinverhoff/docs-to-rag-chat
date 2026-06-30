import inspect
from pathlib import Path

import pytest


def test_gdrive_source_implements_protocol():
    """GoogleDriveSource has a fetch_documents method with dest_dir parameter."""
    from pipeline.sources.gdrive import GoogleDriveSource
    assert hasattr(GoogleDriveSource, "fetch_documents")
    sig = inspect.signature(GoogleDriveSource.fetch_documents)
    assert "dest_dir" in sig.parameters
