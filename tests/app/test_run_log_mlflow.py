from types import SimpleNamespace

import mlflow

from text2motion.app.run_log import MLFLOW_RUN_ID_FILE, log_metrics, start_run


def test_run_without_the_flag_makes_no_mlflow_record(tmp_path):
    run_dir = start_run("plain", SimpleNamespace(mlflow_tracking=False), tmp_path)
    log_metrics(run_dir, {"epoch": 1, "loss": 2.0})
    assert not (run_dir / MLFLOW_RUN_ID_FILE).exists()
    assert not (tmp_path / "mlflow.db").exists()


def test_tracked_run_records_metrics_and_continues_after_a_resume(tmp_path):
    config = SimpleNamespace(mlflow_tracking=True)
    run_dir = start_run("tracked", config, tmp_path, {"epochs": 2})
    log_metrics(run_dir, {"epoch": 1, "loss": 2.0, "device": "cpu", "resumed": False})
    run_id = (run_dir / MLFLOW_RUN_ID_FILE).read_text(encoding="utf-8")
    mlflow.end_run()
    start_run("tracked", config, tmp_path, existing_run_dir=run_dir)
    log_metrics(run_dir, {"epoch": 2, "loss": 1.5})
    mlflow.end_run()
    assert (run_dir / MLFLOW_RUN_ID_FILE).read_text(encoding="utf-8") == run_id
    client = mlflow.MlflowClient()
    history = client.get_metric_history(run_id, "loss")
    assert [(point.step, point.value) for point in history] == [(1, 2.0), (2, 1.5)]
    assert client.get_run(run_id).data.params == {"epochs": "2"}
    assert "resumed" not in client.get_run(run_id).data.metrics
