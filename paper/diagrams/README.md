# Architecture diagrams (Graphviz / DOT)

Seven granular maps — each scoped to one subsystem so it carries ~3 minutes of talk
(together ≈ a 20-minute walkthrough). One node = one function/idea; color groups the
sub-components of a single module.

| File | Covers |
|---|---|
| `data-pipeline.dot` | raw AMASS → SMPL-X forward → `process_file` → normalization → two corpora |
| `training-data.dot` | how a batch is built: indexing, `__getitem__`, augmentation, `collate_pad` |
| `training-loop.dot` | one generator training step: targets, pkeep, CFG, forward, every loss term, optimize |
| `tokenizer.dot` | Contribution A: encoder → Grouped-FSQ → decoder → recon result vs RVQ |
| `generator-twins.dot` | Contribution B: shared front-end → Transformer vs Mamba internals → sampling |
| `evaluation.dot` | split discipline → Guo evaluator → each metric → comparability rules |
| `demo.dot` | prompt → stream → rot6d→SMPL-X (no IK) → 60 fps skinned render |

## Preview (VSCode "Graphviz Interactive Preview" extension)

Open any `.dot` file, then `Ctrl+Shift+P` → **"Graphviz Interactive Preview: Preview
Graphviz / Dot (beside)"** (or the graph icon in the editor title bar). Click a node to
trace its edges; export to SVG from the panel for the dissertation.
