## Epics

The status of each task is in `.claude/docs/epic_tasks.csv`, one line per task. This section holds the plan.

### A0 -- The agent system (approved 2026-11-04, running)
- **Question:** can Copilot agents plan, act, check and report with no manual step after the plan approval?
- **Depends on:** none. **Time box:** 2026-11-08 (A5: 3 nights). **Budget:** about 6 Copilot calls, 0 GPU hours.
- **Debate:** Google Scholar has no official API, so the choice is Semantic Scholar and arXiv. Self-critique needs a tool check (Huang et al. 2023).
- **Tasks:**
  - T0 Disable `understanding_check.py`. Done.
  - T1 ReAct, idea and project map rules in `.github/copilot-instructions.md`. Done.
  - T2 This template. Done.
  - T3-T8 Six agent files in `.github/agents/`, with patterns. Done.
  - T9 Move `.claude/agents/*.md` to `.claude/_archive/agents/` (`.claude/` is not in git, so no delete). Done.
  - T10 Skill: `--agent`, graph query file, 2 branches. Done.
  - T11 Code graph rebuild without `third_party/` (`.graphifyignore`; backup in `graphify-out/2026-11-04/`). Done: 2961 nodes, 0 from `third_party`, 0 from missing files.
  - T12 One test call. Done when the report has 4 sections in order. Done: first call failed (no web access, brief cut); fixed with `--allow-url` and a brief file. Sol and terra both found that `bib.bib:29-36` cites CVPR with the arXiv authors and title. Models per role decided.
  - T13-T15 Paper watch: script, Copilot marks, 03:00 task. Done 2026-11-04: `scripts/research/paper_watch.py` (arXiv + OpenAlex; Semantic Scholar gave HTTP 429 with no key), first report 42 papers; luna marks 13 to compare, 24 relevant, 5 not relevant, novelty threats none; task `ThesisPaperWatch` daily 03:00. Open: the A5 criterion of 3 nights in sequence.
  - T16 Epic loop rules in the `copilot-subagent` skill. Done.
  - T17 One test epic: epic A1 is the test of the epic loop (owner decision 2026-11-04).

### A1 -- Compact motion and text control (approved 2026-11-04, running)
- **Question:** can a more compact motion form or a preset vocabulary cut the training cost and keep the text in control over long streams, inside the thesis time?
- **Use cases:** a director prompt starts a continuous stream; the prompt changes in the stream (a transition); a long stream where the text must stay in control; a preset library (walk, sit, turn) with blends.
- **Edge cases:** contradictory prompts; an action not in the data; a transition between very different poses; foot contact during a transition.
- **Depends on:** A0 (agents), phase 4 (the retrain). The tokenizer is Contribution A and frozen: a change needs a new ADR and both twins retrained.
- **Time box:** 2026-11-10. **Budget:** about 20 Copilot calls (about 200 credits), about 2 GPU hours of benchmarks, 0 training runs.
- **Ideas (2 tests, 1 prediction each, low confidence):**
  - Compact form trains faster. Tests: recon-FID against tokens per second from the tokenizer matrix; step time at half length with random tokens. Prediction: half the positions gives a Mamba step about 2x faster and recon-FID above 0.03.
  - Presets with natural transitions. Tests: share of the training token corpus covered by the top 1,000 segments; joins of 2 clips (hard cut, linear, minimum jerk, generator) scored by jerk and foot slide. Prediction: coverage below 30%.
  - Keep the text active. Tests: Guo matcher R@1 per 10-second window of a long stream; prefix only against the lesson 20 restart. Prediction: R@1 falls with time, faster for Mamba with no restart.
- **Tasks:**
  - R1 The 2 "compact" tests (eval-metrics, sol).
  - R2 Survey of compact motion forms with HumanML3D numbers (scout terra + architect sol).
  - R3 Survey of preset and transition methods that keep causal streaming (scout terra + render sol).
  - R4 Survey of biomechanics for natural transitions and how to measure "natural" (scout terra + data sol).
  - R5 The 2 "text" tests, from lesson 20 (architect sol + eval sol).
  - R6 The 2 "preset" tests (data sol + eval sol).
  - R7 Decision table and debate with the owner.
- **Finish criterion:** a decision table with at least 3 options; each has a verified source, one measured number from R1, R5 or R6, a cost in GPU hours, and its effect on the citable protocol. The owner chooses: phase 4, future work, or drop.

