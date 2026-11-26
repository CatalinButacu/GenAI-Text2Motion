# python scripts/maintenance/run_epic.py --epic A10 --dry-run
import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from types import MappingProxyType

REPO = Path(__file__).resolve().parents[2]
TASKS_CSV = REPO / "docs" / "project" / "epic_tasks.csv"
EPICS_CSV = REPO / "docs" / "project" / "epics.csv"
LOG_DIR = REPO / "runs" / "logs" / "epic_runner"
STOP_FILE = LOG_DIR / "STOP"
CSV_PATHS = ("docs/project/epic_tasks.csv", "docs/project/epics.csv")
TASK_ENV = MappingProxyType({"RUNNER_SESSION": "1"})
CSV_ONLY_ENV = MappingProxyType({"RUNNER_SESSION": "1", "STOP_GATE": "off"})
RUNNABLE_EPIC_STATES = frozenset({"approved", "running"})
OWNER_AGENTS = frozenset({"owner", "gpu"})
PLANNED_EPIC_STATE = "planning"
PERMISSION_MODES = ("auto", "acceptEdits", "dontAsk", "bypassPermissions")
EPIC_ID = re.compile(r"^A(\d+)$")
TASK_KEY = ("epic", "id")
EPIC_KEY = ("id",)
TASK_FIELDS = frozenset({"status", "evidence", "updated"})
REPLAN_FIELDS = frozenset(
    {"status", "evidence", "updated", "task", "finish_criterion", "depends_on"}
)
TASK_PROMPT = """Do exactly one task: {id} of epic {epic} in docs/project/epic_tasks.csv.
Task: {task}
Finish criterion: {finish_criterion}

Rules:
- Follow AGENTS.md and CLAUDE.md. Work on this task only.
- Write short sentences. Report your work as numbered steps: step, reason, result.
- Read the code graph section at the end first. For more, run `graphify query "<topic>" --budget 1500` before you search code.
- Do not start GPU jobs, training, downloads or any public action.
- Tests that already fail before your change are not yours: do not edit, skip or revert anything to make them pass, and do not touch tests/golden or the module ledger. Name them in evidence.
- In epic_tasks.csv, edit only the status, evidence and updated fields of row {id}. Do not edit epics.csv.
- When the finish criterion is met with tool evidence: set status to done, write the measured evidence, then commit small.
- Commit: run `git status --short` first. Stage only files you created, or files that were clean before you changed them, with `git add <path>`. Never stage a file that had uncommitted changes before you started (STATUS.md and the CSVs often do): leave it unstaged. Keep each file's line endings as they were.
- When you cannot meet it, or it needs an owner decision: set status to blocked and write the reason in evidence.
- Then stop."""
REPLAN_PROMPT = """Task {id} of epic {epic} in docs/project/epic_tasks.csv is blocked.
Task: {task}
Finish criterion: {finish_criterion}
Reason in its evidence: {evidence}

- Write short sentences. Report your work as numbered steps: step, reason, result.
- Read the code graph section at the end first. For more, run `graphify query "<topic>" --budget 1500` before you search code.
Decide which case applies, do only that, then stop. Do not do the task itself.
A) The blocker needs the owner: a decision, a login, the supervisor, a GPU job, money, a public action,
   or a change to a thesis claim (AGENTS.md step 7 stop points). Leave status blocked.
   Start evidence with "OWNER:" and one short question.
B) Another approach can remove the blocker. Add new task rows to epic {epic}: new unique ids, status open,
   a measurable finish_criterion, depends_on as a ';' list. In row {id} you may change task, finish_criterion,
   depends_on, evidence and updated, and set status to open. Prefer the simplest approach that reuses existing code.
Do not change any other row. Do not edit epics.csv. Do not commit."""
GRAPH_SECTION = """

Code graph (`graphify query` on the task text, refreshed before this session):
{graph}"""
GRAPH_ANSWER_CODES = frozenset({0, 4})
PLAN_PROMPT = """Read the owner's direction in {direction}.
Read AGENTS.md, the newest 3 entries of docs/project/STATUS.md, docs/project/epics.csv and docs/project/epic_tasks.csv.
Run `graphify query "<topic>" --budget 1500` for the code areas the direction touches.

- Write short sentences. Report your work as numbered steps: step, reason, result.
Plan one epic that follows the direction:
- Add one row to epics.csv: id {epic_id}, state {state}, the question, depends_on, time_box, budget,
  a measurable finish_criterion, the task ids, updated {today}.
- Add its tasks to epic_tasks.csv: new unique ids, epic {epic_id}, status open, agent, model -, task,
  a measurable finish_criterion, depends_on as a ';' list, updated {today}.
- Small tasks: one session each, with a check a tool can run.
- Where real options exist, add a task that measures two options in one process before the choice (AGENTS.md Experiments).
- Set agent to owner for a task that needs the owner, a login or the supervisor. Set agent to gpu for a GPU job.
- Reuse existing code and past work. Name the STATUS entry or task id it builds on in the task text.
Do not change existing rows. Do not change code. Do not commit. Then stop."""


class RunnerError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    epic: str
    status: str
    agent: str
    task: str
    finish_criterion: str
    evidence: str
    depends_on: str

    def key(self) -> str:
        return f"{self.epic}/{self.id}"


def parse_dependencies(text: str, task_id: str) -> tuple[str, ...]:
    if " or " in text:
        raise RunnerError(
            f"task {task_id}: dependency '{text}' uses 'or'; the runner supports only ';' lists"
        )

    return tuple(part.strip() for part in text.split(";") if part.strip())


def read_rows(path: Path, key_fields: tuple[str, ...]) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    by_key: dict[str, dict[str, str]] = {}

    for row in rows:
        key = "/".join(row[field] for field in key_fields)

        if key in by_key:
            raise RunnerError(f"duplicate key {key} in {path}")

        if None in row or None in row.values():
            raise RunnerError(f"row {key} in {path} does not have one value per header column")

        by_key[key] = {name: value or "" for name, value in row.items()}

    return by_key


def read_tasks(path: Path) -> list[Task]:
    return [
        Task(
            id=row["id"],
            epic=row["epic"],
            status=row["status"].strip(),
            agent=row["agent"].strip(),
            task=row["task"],
            finish_criterion=row["finish_criterion"],
            evidence=row["evidence"],
            depends_on=row["depends_on"],
        )
        for row in read_rows(path, TASK_KEY).values()
    ]


def read_epic_state(path: Path, epic: str) -> str:
    rows = read_rows(path, EPIC_KEY)

    if epic not in rows:
        raise RunnerError(f"epic {epic} is not in {path}")

    return rows[epic]["state"].strip()


def require_runnable_epic(state: str, epic: str) -> None:
    if state in RUNNABLE_EPIC_STATES:
        return

    raise RunnerError(f"epic {epic} has state '{state}'; the owner must approve it first")


def next_epic_id(epic_ids: Sequence[str]) -> str:
    numbers = [int(match.group(1)) for epic_id in epic_ids if (match := EPIC_ID.match(epic_id))]

    return f"A{max(numbers, default=-1) + 1}"


def next_task(tasks: Sequence[Task], epic: str) -> Task | None:
    status_by_id = {task.id: task.status for task in tasks}
    ambiguous = {
        task_id for task_id, count in Counter(task.id for task in tasks).items() if count > 1
    }

    for task in tasks:
        if task.epic != epic or task.status != "open" or task.agent in OWNER_AGENTS:
            continue

        dependencies = parse_dependencies(task.depends_on, task.id)
        missing = [dep for dep in dependencies if dep not in status_by_id]

        if missing:
            raise RunnerError(f"task {task.id}: unknown dependencies {missing}")

        shared = [dep for dep in dependencies if dep in ambiguous]

        if shared:
            raise RunnerError(f"task {task.id}: dependencies {shared} name more than one task")

        if all(status_by_id[dep] == "done" for dep in dependencies):
            return task

    return None


def changed_fields(before: Mapping[str, str], after: Mapping[str, str]) -> set[str]:
    return {
        key for key in before.keys() | after.keys() if before.get(key, "") != after.get(key, "")
    }


def check_changes(
    label: str,
    before: Mapping[str, Mapping[str, str]],
    after: Mapping[str, Mapping[str, str]],
    editable: Mapping[str, frozenset[str]],
    is_allowed_new_row: Callable[[Mapping[str, str]], bool],
) -> list[str]:
    removed = sorted(before.keys() - after.keys())

    if removed:
        raise RunnerError(f"{label}: rows removed {removed}")

    for row_id, old_row in before.items():
        extra = changed_fields(old_row, after[row_id]) - editable.get(row_id, frozenset())

        if extra:
            raise RunnerError(
                f"{label}: row {row_id} changed fields it may not touch: {sorted(extra)}"
            )

    added = [row_id for row_id in after if row_id not in before]
    refused = [row_id for row_id in added if not is_allowed_new_row(after[row_id])]

    if refused:
        raise RunnerError(f"{label}: new rows not allowed here: {refused}")

    return added


def no_new_rows(row: Mapping[str, str]) -> bool:
    return False


@dataclass(frozen=True, slots=True)
class Tables:
    tasks_csv: Path
    epics_csv: Path

    def snapshot(self) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
        return read_rows(self.tasks_csv, TASK_KEY), read_rows(self.epics_csv, EPIC_KEY)


def guard_task_session(tables: Tables, task: Task, session: Callable[[], None]) -> None:
    tasks_before, epics_before = tables.snapshot()
    session()
    tasks_after, epics_after = tables.snapshot()
    label = f"task {task.id}"
    check_changes(
        label, tasks_before, tasks_after, MappingProxyType({task.key(): TASK_FIELDS}), no_new_rows
    )
    check_changes(label, epics_before, epics_after, MappingProxyType({}), no_new_rows)


def guard_replan_session(tables: Tables, task: Task, session: Callable[[], None]) -> list[str]:
    tasks_before, epics_before = tables.snapshot()
    session()
    tasks_after, epics_after = tables.snapshot()
    label = f"replan {task.id}"

    def is_open_row_of_epic(row: Mapping[str, str]) -> bool:
        return row["epic"] == task.epic and row["status"].strip() == "open"

    added = check_changes(
        label,
        tasks_before,
        tasks_after,
        MappingProxyType({task.key(): REPLAN_FIELDS}),
        is_open_row_of_epic,
    )
    check_changes(label, epics_before, epics_after, MappingProxyType({}), no_new_rows)

    return added


def guard_plan_session(tables: Tables, epic_id: str, session: Callable[[], None]) -> list[str]:
    tasks_before, epics_before = tables.snapshot()
    session()
    tasks_after, epics_after = tables.snapshot()
    label = f"plan {epic_id}"

    def is_planned_epic(row: Mapping[str, str]) -> bool:
        return row["id"] == epic_id and row["state"].strip() == PLANNED_EPIC_STATE

    def is_open_task_of_epic(row: Mapping[str, str]) -> bool:
        return row["epic"] == epic_id and row["status"].strip() == "open"

    new_epics = check_changes(
        label, epics_before, epics_after, MappingProxyType({}), is_planned_epic
    )

    if new_epics != [epic_id]:
        raise RunnerError(
            f"{label}: expected one new epic {epic_id} in state {PLANNED_EPIC_STATE}, got {new_epics}"
        )

    new_tasks = check_changes(
        label, tasks_before, tasks_after, MappingProxyType({}), is_open_task_of_epic
    )

    if not new_tasks:
        raise RunnerError(f"{label}: the plan added no tasks")

    return new_tasks


def status_of(path: Path, task_key: str) -> str:
    for task in read_tasks(path):
        if task.key() == task_key:
            return task.status

    raise RunnerError(f"task {task_key} disappeared from {path}")


def owner_tasks(tables: Tables, epic: str) -> list[str]:
    return [
        task.id
        for task in read_tasks(tables.tasks_csv)
        if task.epic == epic and task.status == "open" and task.agent in OWNER_AGENTS
    ]


def run_epic(
    epic: str,
    max_tasks: int,
    tables: Tables,
    work: Callable[[Task], None],
    replan: Callable[[Task], None],
) -> list[str]:
    finished: list[str] = []
    replanned: set[str] = set()

    for _ in range(max_tasks):
        if STOP_FILE.exists():
            print(f"epic {epic}: stop file {STOP_FILE} found; stop before the next task")
            return finished

        task = next_task(read_tasks(tables.tasks_csv), epic)

        if task is None:
            print(
                f"epic {epic}: no runnable task; waiting for the owner: {owner_tasks(tables, epic)}"
            )
            return finished

        print(f"task {task.id}: start")
        guard_task_session(tables, task, lambda task=task: work(task))
        status = status_of(tables.tasks_csv, task.key())

        if status == "done":
            finished.append(task.id)
            continue

        if status != "blocked" or task.id in replanned:
            print(f"task {task.id}: status is '{status}' after the session; stop for the owner")
            return finished

        replanned.add(task.id)
        blocked = next(row for row in read_tasks(tables.tasks_csv) if row.key() == task.key())
        print(f"task {task.id}: blocked; replan")
        added = guard_replan_session(tables, blocked, lambda blocked=blocked: replan(blocked))
        status = status_of(tables.tasks_csv, task.key())

        if status != "open":
            print(f"task {task.id}: replan left it '{status}'; stop for the owner")
            return finished

        print(f"task {task.id}: reopened; new tasks {added}")

    print(f"epic {epic}: reached --max-tasks {max_tasks}")

    return finished


def build_task_prompt(task: Task) -> str:
    return TASK_PROMPT.format(
        id=task.id, epic=task.epic, task=task.task, finish_criterion=task.finish_criterion
    )


def build_replan_prompt(task: Task) -> str:
    return REPLAN_PROMPT.format(
        id=task.id,
        epic=task.epic,
        task=task.task,
        finish_criterion=task.finish_criterion,
        evidence=task.evidence,
    )


def build_plan_prompt(direction: Path, epic_id: str, today: date) -> str:
    return PLAN_PROMPT.format(
        direction=direction.as_posix(),
        epic_id=epic_id,
        state=PLANNED_EPIC_STATE,
        today=today.isoformat(),
    )


def build_command(
    claude: str, prompt: str, permission_mode: str, budget_usd: float, model: str
) -> list[str]:
    return [
        claude,
        "-p",
        prompt,
        "--permission-mode",
        permission_mode,
        "--max-budget-usd",
        str(budget_usd),
        "--model",
        model,
        "--output-format",
        "json",
    ]


def run_session(
    command: list[str], label: str, timeout_s: float, session_env: Mapping[str, str]
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{label}_{stamp}.json"
    env = {**os.environ, **session_env}

    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )

    if result.returncode != 0:
        raise RunnerError(
            f"{label}: claude exited with {result.returncode} ({session_message(log_path)}); log {log_path}"
        )

    print(f"{label}: session finished; log {log_path}")


def session_message(log_path: Path) -> str:
    try:
        record = json.loads(log_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return f"log unreadable: {error}"

    return f"{record.get('api_error') or record.get('subtype')}: {record.get('result')}"


def other_files_fingerprint() -> str:
    excluded = [f":(exclude){path}" for path in CSV_PATHS]
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", ".", *excluded],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", ".", *excluded],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    digest = hashlib.sha256(diff)

    for name in untracked.split(b"\0"):
        if not name:
            continue

        stat = (REPO / name.decode("utf-8")).stat()
        digest.update(name + f"|{stat.st_size}|{stat.st_mtime_ns}".encode())

    return digest.hexdigest()


def run_csv_only_session(command: list[str], label: str, timeout_s: float) -> None:
    before = other_files_fingerprint()
    run_session(command, label, timeout_s, CSV_ONLY_ENV)

    if other_files_fingerprint() != before:
        raise RunnerError(
            f"{label}: the session changed files other than the two CSVs; check git status"
        )


def parse_arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan an epic from a direction, or run its tasks in fresh claude -p sessions."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--epic", help="run the open tasks of this approved epic")
    mode.add_argument("--plan", type=Path, help="plan a new epic from this direction file")
    parser.add_argument("--max-tasks", type=int, default=1)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    parser.add_argument("--timeout-min", type=float, default=60.0)
    parser.add_argument("--permission-mode", choices=PERMISSION_MODES, default="auto")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args(argv)


def refresh_graph() -> None:
    graphify = shutil.which("graphify")

    if graphify is None:
        raise RunnerError("'graphify' is not on PATH; the sessions need a fresh code graph")

    subprocess.run([graphify, "update", "."], cwd=REPO, capture_output=True, check=True)


def graph_context(question: str) -> str:
    graphify = shutil.which("graphify")

    if graphify is None:
        raise RunnerError("'graphify' is not on PATH; the sessions need a fresh code graph")

    result = subprocess.run(
        [graphify, "query", question, "--budget", "1500"],
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode not in GRAPH_ANSWER_CODES:
        raise RunnerError(f"graphify query failed ({result.returncode}): {result.stderr.strip()}")

    return GRAPH_SECTION.format(graph=result.stdout.strip())


def find_claude(name: str) -> str:
    claude = shutil.which(name)

    if claude is None:
        raise RunnerError(f"'{name}' is not on PATH")

    return claude


def plan_main(args: argparse.Namespace, tables: Tables) -> int:
    direction: Path = args.plan

    if not direction.is_file():
        raise RunnerError(f"direction file {direction} does not exist")

    epic_id = next_epic_id(list(read_rows(tables.epics_csv, EPIC_KEY)))
    prompt = build_plan_prompt(direction, epic_id, date.today())

    if args.dry_run:
        print(prompt)
        return 0

    claude = find_claude(args.claude)
    refresh_graph()
    command = build_command(claude, prompt, args.permission_mode, args.budget_usd, args.model)
    new_tasks = guard_plan_session(
        tables,
        epic_id,
        lambda: run_csv_only_session(command, f"plan_{epic_id}", args.timeout_min * 60),
    )
    print(
        f"plan {epic_id}: state {PLANNED_EPIC_STATE}, tasks {new_tasks}. Set its state to approved to run it."
    )

    return 0


def epic_main(args: argparse.Namespace, tables: Tables) -> int:
    require_runnable_epic(read_epic_state(tables.epics_csv, args.epic), args.epic)

    if args.dry_run:
        task = next_task(read_tasks(tables.tasks_csv), args.epic)

        if task is None:
            print(f"epic {args.epic}: no open task with all dependencies done")
            return 0

        print(build_task_prompt(task))
        return 0

    claude = find_claude(args.claude)
    timeout_s = args.timeout_min * 60

    def work(task: Task) -> None:
        refresh_graph()
        prompt = build_task_prompt(task) + graph_context(task.task)
        run_session(
            build_command(claude, prompt, args.permission_mode, args.budget_usd, args.model),
            task.id,
            timeout_s,
            TASK_ENV,
        )

    def replan(task: Task) -> None:
        refresh_graph()
        prompt = build_replan_prompt(task) + graph_context(task.task)
        run_csv_only_session(
            build_command(claude, prompt, args.permission_mode, args.budget_usd, args.model),
            f"replan_{task.id}",
            timeout_s,
        )

    finished = run_epic(args.epic, args.max_tasks, tables, work, replan)
    print(f"done: {finished}")

    return 0


def main(argv: Sequence[str]) -> int:
    args = parse_arguments(argv)
    tables = Tables(TASKS_CSV, EPICS_CSV)

    if args.plan is not None:
        return plan_main(args, tables)

    return epic_main(args, tables)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
