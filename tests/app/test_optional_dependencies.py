import sys
from types import SimpleNamespace

import pytest

from text2motion.app.run_log import start_run
from text2motion.motion.preparation import write_motions


def test_the_converter_names_the_prep_extra_when_pyarrow_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)

    with pytest.raises(ImportError, match="uv sync --extra prep"):
        write_motions(tmp_path, tmp_path)


def test_tracking_names_the_tracking_extra_when_mlflow_is_missing(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)

    with pytest.raises(ImportError, match="uv sync --extra tracking"):
        start_run("tracked", SimpleNamespace(mlflow_tracking=True), tmp_path)
