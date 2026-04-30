# Dataset Expertise and EDA (Grounded, Repository-Specific)

Date: 2026-03-31
Scope: Local datasets in `data/` for the physics-constrained video generation pipeline.

## 1. Evidence Base and Method

This report is grounded in:
- Local dataset inventory and recursive file statistics under `data/`.
- Existing project EDA outputs:
  - `data/master_eda_results.json`
  - `data/deep_analysis_stats.json`
- Existing EDA scripts:
  - `scripts/tools/master_eda.py`
  - `scripts/tools/deep_probe_analysis.py`
- Dataset planning reference:
  - `doc/planning/02_DATASETS.md`

Important methodology note:
- Some reported counters in `master_eda_results.json` are known to undercount certain datasets due to non-recursive globbing in `scripts/tools/master_eda.py` for Inter-X and PAHOI. This report corrects those using direct recursive filesystem EDA.

## 2. Global Dataset Inventory (Observed)

| Dataset | Files (recursive) | Approx Size | Dominant Formats | Operational Role |
|---|---:|---:|---|---|
| AMASS | 16,906 | 170.152 GB | `.npz` | Core SMPL/SMPL-X motion corpus, physics and motion priors |
| HumanML3D | 82,945 | 31.912 GB | `.npy`, `.txt` | Text-motion semantic alignment and motion-language analysis |
| Inter-X | 57,059 | 44.350 GB | `.npz`, `.npy`, `.txt` | Two-person interaction data in SMPL-X style |
| ARCTIC | 1,525 | 3.936 GB | `.npy`, `.obj`, `.json` | Human-object manipulation/object-centric dynamics |
| PAHOI | 6,386 | 5.046 GB | `.npz`, `.txt`, `.fbx`, `.csv`, `.bvh`, `.npy` | Human-object interaction, multi-format sequence data |
| M1 training | 3 | 0.004 GB | `.jsonl` | Parser supervised fine-tuning/evaluation |
| Knowledge base | 51 | 0.091 GB | `.json`, `.faiss` | Retrieval support for understanding/planning |
| t2m | 1 | 0.229 GB | `.tar` | Archived motion/text asset |
| t2m_download | 26 | 1.246 GB | `.txt`, `.npy`, `.tar` | Partial/auxiliary text-motion assets |
| conceptnet_cache | 1 | 0.464 GB | (single cache artifact) | Cached commonsense graph data |

## 3. Per-Dataset Expertise and EDA Conclusions

## 3.1 AMASS (`data/amass`)

### EDA facts
- `master_eda_results.json`:
  - sequences (`.npz`): 16,884
  - total frames: 21,387,899
  - total hours: 198.04 h (frame-rate normalized)
  - gender labels: female 5,821; male 10,586; inferred unknown/other 477
  - hand-active sequences: 16,407
- Recursive inventory confirms 16,884 `.npz` plus support files.

### Interpretation (grounded)
- Scale is sufficient for robust motion prior learning and physics-aware temporal modeling.
- Hand activity ratio is very high: 16,407 / 16,884 = 97.18%, which is favorable for SMPL-X upper-body realism.
- Gender annotation is skewed toward male-labeled sequences:
  - male share among labeled = 10,586 / (10,586 + 5,821) = 64.52%
  - female share among labeled = 35.48%
  - this can bias gait/posture priors if not balanced in training.

### Risks
- Potential demographic/bodily-style bias from label skew.
- Heterogeneous source sets inside AMASS can produce style/domain drift across sub-datasets.

### Best use
- Base pretraining for MotionSSM/PhysicsSSM.
- Physics plausibility priors (contacts, stability, kinematic continuity).

### Truth conclusion
- AMASS is the strongest foundation for general motion realism, but reweighting or stratified sampling is required to reduce demographic and style bias.

## 3.2 HumanML3D (`data/humanml3d`)

### EDA facts
- Recursive inventory:
  - motion files (`motion_data/*.npy`): 26,846
  - caption files (`texts/*.txt`): 29,232
- Existing EDA summary:
  - caption count: 29,232
  - token frequency dominated by locomotion and body-part terms (`walk`, `left`, `right`, `hand`, `arm`).
- Deep probe (`data/deep_analysis_stats.json`, sample size 1000):
  - word_len_corr (word count vs sequence length): 0.2627
  - vigor_vel_corr (vigor heuristic vs root velocity): 0.0193
  - avg_root_vel: 0.9997
  - avg_joint_var: 0.006656
- Split files:
  - train 23,384
  - val 1,338
  - test 4,042

### Interpretation (grounded)
- Caption-to-motion ratio = 29,232 / 26,846 = 1.089 indicates multi-caption supervision per motion (good for language robustness).
- Moderate positive length correlation (0.2627) indicates textual detail scales somewhat with sequence length, but not strongly enough to infer precise temporal complexity.
- Near-zero vigor-velocity correlation (0.0193) implies simple keyword vigor heuristics are insufficient to infer physical intensity from captions.

### Risks
- Overfitting to frequent locomotion verbs (`walk`-heavy semantic prior).
- Weak semantics-to-kinematics coupling if lexical heuristics are used without richer language modeling.

### Best use
- Text conditioning and semantic control layers.
- Retrieval augmentation for caption-grounded generation/evaluation.

### Truth conclusion
- HumanML3D is strong for language coverage and paraphrastic supervision, but intensity semantics require richer modeling than surface-level token heuristics.

## 3.3 Inter-X (`data/inter-x`)

### EDA facts
- Recursive inventory:
  - `.npz`: 22,776
  - `.npy`: 22,778
  - `.txt`: 11,401
- Split files:
  - all 11,388
  - train 9,110
  - val 570
  - test 1,708
- Naming structure:
  - motions: `motions/<clip-id>/P1.npz` and `P2.npz`
  - text: `texts/<clip-id>.txt`
- Consistency pattern:
  - 22,776 motion `.npz` is exactly 2 x 11,388 split entries (two actors per interaction).

### Interpretation (grounded)
- Dataset is structurally coherent for dyadic modeling; explicit P1/P2 decomposition aligns with interaction-centric objectives.
- Slight text/split mismatch (11,401 txt vs 11,388 split all; and 11,386 text files in `texts/`) indicates small indexing noise that should be filtered during preprocessing.

### Risks
- Pairwise synchronization and role assignment errors if clip-ID joins are not strict.
- Split/list mismatch can leak untracked examples into training/evaluation.

### Best use
- Multi-agent interaction modeling, turn-taking, contact-rich human-human sequences.

### Truth conclusion
- Inter-X is the most strategically valuable dataset for two-person interaction realism, provided strict ID-based pairing and split sanitation are enforced.

## 3.4 ARCTIC (`data/arctic`)

### EDA facts
- Recursive inventory:
  - 1,206 `.npy`
  - 54 `.obj`
  - 49 `.json`
  - plus support materials (`.jpg`, `.mtl`, `.txt`, `.md`, scripts)
- `master_eda_results.json` reports `.npy` file_count 1,206 (consistent).

### Interpretation (grounded)
- ARCTIC combines numerical sequences with object meshes and metadata, which is useful for object-aware manipulation scenarios.
- Compared with AMASS/Inter-X scale, ARCTIC is smaller, so it should be used as a specialization or fine-tuning source rather than sole pretraining corpus.

### Risks
- Limited sample count can overfit object categories and grasp styles.
- Domain shift if transferred directly into broad free-motion generation.

### Best use
- Object-centric contact refinement and manipulation validation sets.

### Truth conclusion
- ARCTIC is a high-value specialization dataset for hand-object realism, not a general motion prior replacement.

## 3.5 PAHOI (`data/pahoi`)

### EDA facts
- Recursive inventory:
  - `.csv`: 562
  - `.bvh`: 562
  - `.npy`: 562
  - `.npz`: 2,248
  - `.txt`: 1,294
  - `.fbx`: 1,157
- Pattern: one primary sequence appears to map across csv/bvh/npy equally (562 each), with additional derived or participant/modality-specific `.npz` artifacts (2,248 = 4 x 562).
- `master_eda_results.json` reports PAHOI file_count 0 due non-recursive glob in script (not true for local repository state).

### Interpretation (grounded)
- PAHOI is multi-format and conversion-friendly, ideal for cross-toolchain validation (skeleton vs mesh vs tabular annotations).
- Structural multiplicity suggests rich per-sequence derivatives, useful for representation consistency checks.

### Risks
- Pipeline ingestion complexity due heterogeneous file formats and potentially different coordinate conventions.

### Best use
- Cross-format consistency benchmarks and human-object interaction stress-tests.

### Truth conclusion
- PAHOI is a versatile integration/validation dataset and should be treated as a multimodal benchmark layer rather than a single-format training source.

## 3.6 M1 Training (`data/m1_training`)

### EDA facts
- Files:
  - train.jsonl: 12,000 lines
  - val.jsonl: 1,500 lines
  - test.jsonl: 1,500 lines
- Total: 15,000 JSONL examples (not 40,000 as listed in planning doc).
- Schema (sampled):
  - `input`: extraction prompt sentence
  - `target`: JSON-serialized entities + relations

### Interpretation (grounded)
- Data is directly task-aligned for scene extraction and relation prediction.
- Current local size is medium-scale; enough for fine-tuning but may limit long-tail relation coverage.

### Risks
- Label noise in targets (sample shows occasional inconsistent object mentions) can hurt parser precision.

### Best use
- Supervised parser training with strict schema validation and noise filtering.

### Truth conclusion
- M1 training data is usable and aligned, but local corpus size and annotation quality controls are the key bottlenecks for parser robustness.

## 3.7 Knowledge Base (`data/knowledge_base`)

### EDA facts
- 50 JSON chunks (`objects_*.json`) and 1 FAISS index file.

### Interpretation (grounded)
- Structured for retrieval-augmented grounding in understanding/planning modules.
- Chunked JSON + FAISS is appropriate for scalable nearest-neighbor lookup.

### Risks
- Retrieval quality depends on embedding refresh and object schema consistency.

### Best use
- Candidate object retrieval and semantic enrichment in M1/M2 stages.

### Truth conclusion
- KB is operationally mature for retrieval, but should be evaluated for recall on rare/compound prompts.

## 3.8 t2m and t2m_download

### EDA facts
- `data/t2m`: 1 tar archive.
- `data/t2m_download`: 26 files (16 txt, 6 npy, 4 tar).

### Interpretation (grounded)
- Indicates partial or staged ingestion process.
- Not yet in a clean, production-ready dataset form relative to HumanML3D/AMASS/Inter-X.

### Truth conclusion
- t2m assets are currently auxiliary and should be normalized before being part of core training/evaluation loops.

## 3.9 conceptnet_cache

### EDA facts
- Single large cache artifact (~0.464 GB).

### Interpretation (grounded)
- Useful for offline commonsense lookup acceleration and deterministic runs.

### Truth conclusion
- Treat as infrastructure support data, not as a training corpus.

## 4. Cross-Dataset EDA Synthesis (Grounded Truth)

1. The strongest motion prior source is AMASS by size and temporal breadth, but it carries measurable demographic imbalance.
2. HumanML3D provides strong semantic diversity and multi-caption supervision, but lexical vigor heuristics do not explain kinematic intensity.
3. Inter-X has clean dyadic structure and is the best source for two-person interaction modeling.
4. ARCTIC and PAHOI should be used as specialization and validation layers for human-object realism, not as sole backbone corpora.
5. Planning documentation and local storage diverge in places (notably M1 size and PAHOI/Inter-X count reporting logic), so pipeline decisions should follow measured local EDA rather than static planning values.

## 5. Scenario Strategy Matrix (Actionable)

## Scenario S1: Single-person physically plausible generation
- Goal: Stable locomotion and posture with low artifact rate.
- Primary datasets: AMASS + HumanML3D.
- Data to extract/use:
  - AMASS sequence duration and fps-normalized sampling.
  - HumanML3D caption-motion pairing for semantic conditioning.
- Mentions to include:
  - AMASS 198.04 h and 21.39M frames as prior strength.
  - HumanML3D multi-caption ratio (1.089) to justify text robustness.

## Scenario S2: Two-person interaction generation
- Goal: Contact-aware dyadic behavior and role consistency.
- Primary dataset: Inter-X.
- Data to extract/use:
  - Clip-ID joins between `texts/<id>.txt` and `motions/<id>/P1|P2.npz`.
  - Split-constrained loading (train/val/test) to prevent leakage.
- Mentions to include:
  - Exact 2x relation between split count and motion files supports two-actor structure.

## Scenario S3: Human-object manipulation realism
- Goal: Better hand-object/contact plausibility.
- Primary datasets: ARCTIC + PAHOI.
- Data to extract/use:
  - Object mesh metadata (`.obj`, `.json`, `.mtl`) from ARCTIC.
  - Cross-format sequence consistency (csv/bvh/npy/npz) in PAHOI.
- Mentions to include:
  - ARCTIC as specialization set; PAHOI as cross-format benchmark.

## Scenario S4: Parser robustness and scene understanding
- Goal: Better entity/relation extraction and long-tail prompt handling.
- Primary datasets: M1 training + Knowledge Base + conceptnet_cache.
- Data to extract/use:
  - JSON schema validation errors in M1 labels.
  - Retrieval hit quality from FAISS-backed KB.
- Mentions to include:
  - Current local M1 corpus is 15k examples; prioritize quality over quantity first.

## Scenario S5: Evaluation and ablation design
- Goal: Clear module-level attribution in dissertation experiments.
- Dataset layout:
  - AMASS baseline prior.
  - +HumanML3D semantic conditioning.
  - +Inter-X interaction finetune.
  - +ARCTIC/PAHOI object-contact refinement.
- Mentions to include:
  - Incremental gains should be reported per stage to avoid confounding effects.

## 6. Useful Data Mentions (Ready to Reuse in Writing)

- AMASS contributes 16,884 core `.npz` sequences and ~198.04 hours of motion.
- AMASS hand-active coverage is 97.18% of sequences.
- HumanML3D contains 29,232 captions over 26,846 motion arrays (multi-caption supervision).
- HumanML3D deep probe shows moderate word-count to length correlation (0.2627) but near-zero vigor-to-velocity correlation (0.0193).
- Inter-X split cardinality is 11,388 clips with two motion streams per clip (22,776 `.npz`).
- ARCTIC offers object-conditioned assets (`.obj`/`.json`) plus 1,206 `.npy` motion-related arrays.
- PAHOI exposes strongly aligned multimodal counts: 562 `.csv`, 562 `.bvh`, 562 `.npy`, with expanded `.npz` derivatives.
- M1 parser dataset in local repository is 15,000 JSONL examples.
- Knowledge base uses 50 JSON shards plus FAISS retrieval index.

## 7. Reliability Notes and Next EDA Improvements

- `scripts/tools/master_eda.py` should be corrected to recursive globbing for Inter-X and PAHOI (`rglob`) to avoid false zero counts.
- Add explicit schema audits:
  - missing text-motion pairs,
  - duplicate IDs,
  - invalid JSON target structures in M1 training.
- Add stratified statistics (duration bins, action class bins, interaction type bins) for dissertation-grade validity analysis.

---

This document intentionally separates measured local facts from interpretation and strategy, so all conclusions remain traceable to reproducible EDA evidence.
