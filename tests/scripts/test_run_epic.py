import csv
import importlib.util
import sys
from pathlib import Path

import pytest

RUNNER_SCRIPT = Path("scripts/maintenance/run_epic.py")
TASK_FIELDS: tuple[str, ...] = (
    "id",
    "epic",
    "status",
    "agent",
    "model",
    "task",
    "finish_criterion",
    "evidence",
    "depends_on",
    "updated",
)
EPIC_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "state",
    "question",
    "depends_on",
    "time_box",
    "budget",
    "finish_criterion",
    "tasks",
    "updated",
)


def runner_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_epic", RUNNER_SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def write_rows(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def task_row(
    task_id: str, epic: str, status: str, depends_on: str = "", agent: str = "me"
) -> dict[str, str]:
    return {
        "id": task_id,
        "epic": epic,
        "status": status,
        "agent": agent,
        "task": f"do {task_id}",
        "finish_criterion": "ok",
        "depends_on": depends_on,
    }


def make_tables(runner, tmp_path: Path, tasks: list[dict[str, str]], epic_state: str = "approved"):
    tasks_csv = tmp_path / "tasks.csv"
    epics_csv = tmp_path / "epics.csv"
    write_rows(tasks_csv, TASK_FIELDS, tasks)
    write_rows(
        epics_csv, EPIC_FIELDS, [{"id": "A1", "state": epic_state}, {"id": "A7", "state": "done"}]
    )
    return runner.Tables(tasks_csv, epics_csv)


def edit_task(path: Path, task_id: str, **fields: str) -> None:
    rows = read_rows(path)

    for row in rows:
        if row["id"] == task_id:
            row.update(fields)

    write_rows(path, TASK_FIELDS, rows)


def add_task(path: Path, row: dict[str, str]) -> None:
    write_rows(path, TASK_FIELDS, [*read_rows(path), row])


def no_replan(task) -> None:
    raise AssertionError(f"replan was not expected for {task.id}")


def test_the_next_task_is_open_with_every_dependency_done(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner,
        tmp_path,
        [
            task_row("T1", "A1", "done"),
            task_row("T2", "A1", "open", "T3"),
            task_row("T3", "A1", "open", "T1"),
            task_row("T4", "A2", "open"),
        ],
    )

    task = runner.next_task(runner.read_tasks(tables.tasks_csv), "A1")

    assert task is not None
    assert task.id == "T3"


def test_an_or_dependency_fails_loud_only_in_the_chosen_epic(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("N8", "A2", "open", "N7 or N4")]
    )
    tasks = runner.read_tasks(tables.tasks_csv)

    assert runner.next_task(tasks, "A1").id == "T1"

    with pytest.raises(runner.RunnerError, match="'or'"):
        runner.next_task(tasks, "A2")


def test_a_task_id_used_in_two_epics_is_read_and_fails_only_as_a_dependency(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner,
        tmp_path,
        [
            task_row("T3", "A0", "done"),
            task_row("T3", "A4", "done"),
            task_row("T9", "A1", "open", "T3"),
        ],
    )

    with pytest.raises(runner.RunnerError, match="more than one task"):
        runner.next_task(runner.read_tasks(tables.tasks_csv), "A1")


def test_an_unknown_dependency_fails_loud(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open", "Z9")])

    with pytest.raises(runner.RunnerError, match="unknown dependencies"):
        runner.next_task(runner.read_tasks(tables.tasks_csv), "A1")


def test_an_epic_that_is_not_approved_is_refused(monkeypatch):
    runner = runner_module(monkeypatch)

    runner.require_runnable_epic("running", "E")
    runner.require_runnable_epic("approved", "E")

    with pytest.raises(runner.RunnerError, match="approve"):
        runner.require_runnable_epic("planning", "E")


def test_the_next_epic_id_follows_the_highest_number(monkeypatch):
    runner = runner_module(monkeypatch)

    assert runner.next_epic_id(["A0", "A27", "A9", "X1"]) == "A28"
    assert runner.next_epic_id([]) == "A0"


def test_the_loop_runs_tasks_until_none_is_ready(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("T2", "A1", "open", "T1")]
    )

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="done", evidence="measured")

    assert runner.run_epic("A1", 5, tables, work, no_replan) == ["T1", "T2"]


def test_the_loop_stops_at_max_tasks(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("T2", "A1", "open")]
    )

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="done")

    assert runner.run_epic("A1", 1, tables, work, no_replan) == ["T1"]


def test_owner_tasks_are_skipped_and_the_runnable_tasks_still_run(monkeypatch, tmp_path, capsys):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner,
        tmp_path,
        [
            task_row("T1", "A1", "open", agent="owner"),
            task_row("T2", "A1", "open"),
            task_row("T3", "A1", "open", "T1"),
            task_row("T4", "A1", "open", agent="gpu"),
        ],
    )

    def work(task):
        assert task.agent not in {"owner", "gpu"}
        edit_task(tables.tasks_csv, task.id, status="done")

    assert runner.run_epic("A1", 5, tables, work, no_replan) == ["T2"]
    assert "waiting for the owner: ['T1', 'T4']" in capsys.readouterr().out


def test_a_blocked_task_is_replanned_once_and_the_new_task_runs_first(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open")])
    order: list[str] = []

    def work(task):
        order.append(task.id)

        if task.id == "T1" and order.count("T1") == 1:
            edit_task(tables.tasks_csv, "T1", status="blocked", evidence="missing helper")
            return

        edit_task(tables.tasks_csv, task.id, status="done")

    def replan(task):
        assert task.evidence == "missing helper"
        add_task(tables.tasks_csv, task_row("T1a", "A1", "open"))
        edit_task(tables.tasks_csv, "T1", status="open", depends_on="T1a")

    assert runner.run_epic("A1", 5, tables, work, replan) == ["T1a", "T1"]
    assert order == ["T1", "T1a", "T1"]


def test_a_replan_that_leaves_the_task_blocked_stops_for_the_owner(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("T2", "A1", "open")]
    )

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="blocked", evidence="needs login")

    def replan(task):
        edit_task(tables.tasks_csv, task.id, evidence="OWNER: log in to AMASS?")

    assert runner.run_epic("A1", 5, tables, work, replan) == []
    assert read_rows(tables.tasks_csv)[0]["evidence"].startswith("OWNER:")


def test_a_task_blocked_twice_stops_for_the_owner(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open")])
    replans: list[str] = []

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="blocked")

    def replan(task):
        replans.append(task.id)
        edit_task(tables.tasks_csv, task.id, status="open")

    assert runner.run_epic("A1", 5, tables, work, replan) == []
    assert replans == ["T1"]


def test_the_guard_refuses_a_task_session_that_edits_another_row(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("T2", "A1", "open")]
    )

    def work(task):
        edit_task(tables.tasks_csv, "T2", status="done")

    with pytest.raises(runner.RunnerError, match="row A1/T2 changed fields"):
        runner.run_epic("A1", 5, tables, work, no_replan)


def test_the_guard_refuses_a_task_session_that_edits_its_finish_criterion(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open")])

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="done", finish_criterion="easier")

    with pytest.raises(runner.RunnerError, match="finish_criterion"):
        runner.run_epic("A1", 5, tables, work, no_replan)


def test_the_guard_refuses_a_replan_that_adds_a_task_to_another_epic(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open")])

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="blocked")

    def replan(task):
        add_task(tables.tasks_csv, task_row("X1", "A7", "open"))
        edit_task(tables.tasks_csv, task.id, status="open")

    with pytest.raises(runner.RunnerError, match="new rows not allowed"):
        runner.run_epic("A1", 5, tables, work, replan)


def test_a_plan_session_adds_one_planning_epic_with_open_tasks(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "done")])

    def plan():
        write_rows(
            tables.epics_csv,
            EPIC_FIELDS,
            [*read_rows(tables.epics_csv), {"id": "A8", "state": "planning"}],
        )
        add_task(tables.tasks_csv, task_row("P1", "A8", "open"))
        add_task(tables.tasks_csv, task_row("P2", "A8", "open", "P1"))

    assert runner.guard_plan_session(tables, "A8", plan) == ["A8/P1", "A8/P2"]


def test_the_guard_refuses_a_plan_that_approves_its_own_epic(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "done")])

    def plan():
        write_rows(
            tables.epics_csv,
            EPIC_FIELDS,
            [*read_rows(tables.epics_csv), {"id": "A8", "state": "approved"}],
        )
        add_task(tables.tasks_csv, task_row("P1", "A8", "open"))

    with pytest.raises(runner.RunnerError, match="new rows not allowed"):
        runner.guard_plan_session(tables, "A8", plan)


def test_the_guard_refuses_a_plan_with_no_tasks(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "done")])

    def plan():
        write_rows(
            tables.epics_csv,
            EPIC_FIELDS,
            [*read_rows(tables.epics_csv), {"id": "A8", "state": "planning"}],
        )

    with pytest.raises(runner.RunnerError, match="no tasks"):
        runner.guard_plan_session(tables, "A8", plan)


def test_the_command_carries_the_budget_and_permission_mode(monkeypatch):
    runner = runner_module(monkeypatch)

    command = runner.build_command("claude", "p", "auto", 2.5, "sonnet")

    assert command[:3] == ["claude", "-p", "p"]
    assert command[command.index("--permission-mode") + 1] == "auto"
    assert command[command.index("--max-budget-usd") + 1] == "2.5"
    assert command[command.index("--model") + 1] == "sonnet"


def test_a_stop_file_ends_the_run_before_the_next_task(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(
        runner, tmp_path, [task_row("T1", "A1", "open"), task_row("T2", "A1", "open")]
    )
    stop_file = tmp_path / "STOP"
    monkeypatch.setattr(runner, "STOP_FILE", stop_file)

    def work(task):
        edit_task(tables.tasks_csv, task.id, status="done")
        stop_file.write_text("stop", encoding="utf-8")

    assert runner.run_epic("A1", 5, tables, work, no_replan) == ["T1"]


def test_a_row_with_an_extra_field_fails_loud(monkeypatch, tmp_path):
    runner = runner_module(monkeypatch)
    tables = make_tables(runner, tmp_path, [task_row("T1", "A1", "open")])

    def work(task):
        text = tables.tasks_csv.read_text(encoding="utf-8").rstrip("\n")
        tables.tasks_csv.write_text(text + ",extra\n", encoding="utf-8")

    with pytest.raises(runner.RunnerError, match="one value per header column"):
        runner.run_epic("A1", 5, tables, work, no_replan)
