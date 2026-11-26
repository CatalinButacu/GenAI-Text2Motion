# How we work in this repo (orchestrator flow)

## Roles

- Orchestrator (one agent with shell access): plans, writes briefs, starts jobs, checks every claim with a tool, edits code, commits. It never trusts a worker report alone.
- Workers: Copilot CLI agents in `.github/agents/*.agent.md` (data, architect, training, eval, render, gpu-kernel, research-scout). Shared rules: `.github/copilot-instructions.md`. Default model: `gpt-6.2-sol` for every role. No Claude models, Claude CLI, or Claude subagents in Copilot workers. Ask the owner before using a stronger model. The internal council is the exception: it uses Claude subagents and official skills, as defined in `~/.claude/council/internal.md`.
- Owner: approves each epic plan once and decides at the stop points.
- Workers explore and propose. They do not act. Inside an approved epic, the orchestrator acts without asking and stops only at the step 7 stop points. Rules: `## Agents` in `~/.claude/CLAUDE.md`.

## Where the state is

- Epics: `docs/project/epics.csv`. Tasks: `docs/project/epic_tasks.csv`. One line each: id, epic, status, agent, model, task, finish_criterion, evidence, depends_on, updated.
- History: `docs/project/STATUS.md`, newest entry on top.
- `runs/` is git-ignored, so search tools skip it. Open its files by their full path. Each paper figure has a tracked CSV next to it in `docs/paper/pics/`.

## The unit of work: an epic

- An epic has a name, a question, use cases, a time box, a budget (GPU hours, Copilot calls) and tasks with a measurable finish criterion.
- The owner approves an epic once. Then it runs task by task with no more approvals.

## The loop for each task

1. Plan: write a brief file (task, file paths, finish criterion, maximum words) in a temp folder, not in the repo.
2. Context: `graphify query "<topic>" --budget 1500 > graph.md`, and name that file in the brief.
3. Workers: for a decision, run 2 branches at the same time with 2 different roles:
   `copilot -C <repo> --agent <role> -p "Read the file <brief.md> completely and do the task it describes." -s --model <id> --allow-all-tools --deny-tool shell --deny-tool write --allow-url <domain>`
4. Verify: open every `file:line` and URL that a report cites, and run its numbers again. A disagreement between the 2 branches is the first thing to check.
5. Act: the orchestrator edits the code, runs `pytest`, `ruff`, `pyright` and `scripts/maintenance/check_problems.py`, and commits small in the owner's name.
6. Record: update the task line in `epic_tasks.csv` with the measured evidence.
7. Go to the next task. Stop and ask the owner only for: a plan change, a discovery that changes a thesis claim, a criterion that fails 2 times, a budget overrun, a push or other public action, a new epic.

## Unattended runs (`scripts/maintenance/run_epic.py`)

1. The owner writes a direction from `docs/project/directions/TEMPLATE.md`.
2. `run_epic.py --plan <direction.md>`: one fresh session adds an epic in state `planning` with open tasks.
3. The owner reads the plan and sets the epic state to `approved`. This is the only approval.
4. `run_epic.py --epic <id> --max-tasks N`: one fresh session for each task. A blocked task gets one replan session; it reopens the task with new tasks, or leaves it blocked with an `OWNER:` question.
5. The runner diffs both CSVs after each session and stops on any edit outside that session's rights. Tasks with agent `owner` or `gpu` stop the run.

## Experiments

- Decide with a two-sided test: measure both options in one process before a choice.
- Run a long GPU job as a detached process (`Start-Process`, a pid file, a log file), never as a child of the chat session. Run one GPU job at a time. RAM is tight (16 GB).
- Each new training run: overfit gate first, seed 2026 only, both twins changed together, selection on validation only, clean validation CE every 5 epochs, stop after 10 epochs without a lower validation CE (`--val_ce_patience 10`, evaluation every 5 epochs); five retries are allowed, ties count, and an improvement resets patience. Preserve the counter on resume; `--fresh_selection` explicitly resets selection. Keep completed experiments' original recipes unchanged. Runs before 2026-11-23 are historical and do not enter the paper.
- Benchmarks must use the training precision (TF32 by default).

## Big decisions

- Run the external council (Copilot, no Claude calls): see `~/.claude/council/external.md`. Three `gpt-6.2-sol` members and one `gemini-4-argon` member. 4 calls, no review step, 300 words per answer, no retries.

## Code rules

- No comments, except the class separator box in `src` and the run command on the first line of a script.
- Function names have no underscore prefix. One objective per function. At most 10 files per package.
- Module constants are UPPER_CASE and read-only (tuple, frozenset, MappingProxyType).
- Fail loud. No silent fallbacks.
- Rename only with `scripts/maintenance/rename_everywhere.py`.
- Never `cd` into a subfolder. Use absolute paths or `git -C`.
- Commits: small, in the owner's name, no co-author line. Never push. Never rewrite history.
- Before `uv add`, use `--no-sync`: a sync deletes the compiled Mamba kernel, which is not in `uv.lock`.

## Why the check step matters

A worker report is a hypothesis, not a result. In this project, tool checks found a wrong DOI, an out-of-date claim in a brief (a resume bug that froze the Mamba weights at epoch 30), a wrong precision assumption (training runs TF32), and a unit mix-up. Each one came out because a second source or a tool contradicted the first.
