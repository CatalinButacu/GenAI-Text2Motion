# Direction: <short name>

Copy this file, fill each line, then run:
`.\.venv\Scripts\python.exe scripts/maintenance/run_epic.py --plan docs/project/directions/<file>.md`

- Goal: <the outcome you want, in one sentence>
- Measure: <the number or check that says the goal is met, for example "FID on val below 0.5" or "make check green">
- Limits: <what must not change, for example "no generator retrain", "keep the twins matched">
- Budget: <GPU hours, money per session, time box>
- Known context: <STATUS entry, epic or task ids this builds on; optional>
- Open choices: <options you already see; the planner adds a measured comparison for each; optional>
