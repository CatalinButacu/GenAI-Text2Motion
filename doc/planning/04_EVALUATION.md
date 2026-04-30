# Evaluation & Benchmarks

Metrics, targets, and benchmark strategy for all modules.

---

## Per-Module Metrics

### M1 -- Scene Parser

| Metric | Description | Current | Target |
|--------|-------------|---------|--------|
| BLEU-4 | N-gram overlap | 0.38 | >0.40 |
| ROUGE-L | LCS F1 | 0.52 | >0.55 |
| Entity F1 | Entity extraction P/R/F1 | 0.83 | >0.85 |
| JSON syntax rate | Valid JSON output | ~95% | >97% |
| Inference latency | Per prompt (GPU) | ~200ms | <500ms |

### M4 -- Motion Generator

| Metric | Description | Target |
|--------|-------------|--------|
| FID | Frechet distance on motion features | <15.0 |
| R-Precision (top-3) | Text->motion retrieval accuracy | >0.60 |
| Diversity | Variance across generated samples | >5.0 |

### M4 -- PhysicsSSM (Novel)

| Metric | Description | Target |
|--------|-------------|--------|
| Foot sliding ratio | % frames with feet sliding | <5% |
| Ground penetration | % frames with pelvis below ground | <1% |
| Jerk ratio | Generated / ground truth jerk | <1.2 |
| Gate contribution | Ablation: with vs without gate | Statistically significant |

### M5 -- Physics Simulator

| Metric | Description | Target |
|--------|-------------|--------|
| Penetration count | Objects passing through surfaces | 0 |
| Simulation stability | % runs without NaN / crash | >99% |

### M6 -- Render Engine

| Metric | Description | Target |
|--------|-------------|--------|
| Joint RMSE | Projected vs ground truth (px) | <10 px |

---

## Benchmark Scripts

| Script | Module | Tests |
|--------|--------|-------|
| `tests/benchmark_m1.py` | M1 | Parser accuracy, entity extraction |
| `tests/benchmark_m2.py` | M2 | Planner constraint satisfaction |
| `tests/benchmark_m3.py` | M3 | Asset generation quality |
| `tests/benchmark_m4.py` | M4 | Motion generation metrics |
| `tests/benchmark_m5.py` | M5 | Physics simulation stability |

---

## Ablation Studies (Planned)

1. **PhysicsSSM gate**: Compare full model vs. no-gate (direct concatenation) vs. no-physics (MotionSSM only)
2. **Physics loss lambdas**: Sweep lambda_slide, lambda_pen, lambda_jerk to find Pareto-optimal trade-off
3. **Retrieval cascade**: Measure quality contribution of each retrieval tier (semantic -> SSM -> keyword)

---

## Baselines for Comparison

| System | Type | Metrics Available |
|--------|------|-------------------|
| MDM (Tevet et al.) | Diffusion on motion | FID=0.544, R-Prec=0.611 |
| T2M-GPT (Zhang et al.) | VQ-VAE + GPT | FID=0.116, R-Prec=0.775 |
| MotionDiffuse (Zhang et al.) | Body-part diffusion | FID=0.630 |
