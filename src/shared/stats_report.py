from __future__ import annotations

import math

import numpy as np

from src.shared.constants import OUTLIER_SIGMA, RNG_SEED, T_CRITICAL, Z_95


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = RNG_SEED,
) -> tuple[float, float]:
    """Non-parametric bootstrap confidence interval for the mean.

    Parameters
    ----------
    values:
        Sample of scalar measurements (e.g. per-prompt combined scores).
    n_boot:
        Number of bootstrap resamples.
    alpha:
        Significance level; 0.05 gives a 95% CI.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (ci_low, ci_high) — the (alpha/2, 1-alpha/2) percentiles of the
    bootstrap distribution of the mean.  Returns (nan, nan) for empty input.
    """
    if not values:
        return float("nan"), float("nan")

    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)

    ci_low = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return ci_low, ci_high


def t_critical(n: int) -> float:
    """Return the two-tailed 95% t critical value for n observations."""
    if n <= 1:
        return float("nan")
    df = n - 1
    return T_CRITICAL.get(df, Z_95)


def extract_numeric_values(values: list) -> list[float]:
    result = []

    for v in values:
        try:
            fv = float(v)
            if math.isfinite(fv):
                result.append(fv)
        except (TypeError, ValueError):
            pass

    return result


def build_seed_summary(numeric_values: list[float]) -> dict:
    arr = np.array(numeric_values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else float("nan")
    t = t_critical(n)
    half_width = t * std / math.sqrt(n) if (n > 1 and math.isfinite(std)) else float("nan")

    return {
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width if math.isfinite(half_width) else float("nan"),
        "ci95_high": mean + half_width if math.isfinite(half_width) else float("nan"),
        "n": n,
    }


def aggregate_seeds(per_seed_results: list[dict]) -> dict:
    """Aggregate per-seed result dicts into mean ± std ± 95% CI.

    Numeric keys are replaced by {"mean", "std", "ci95_low", "ci95_high"}.
    Non-numeric keys are passed through from the first seed result.

    Parameters
    ----------
    per_seed_results:
        List of result dicts, one per seed. All dicts must share the same keys.

    Returns
    -------
    dict with same keys; numeric values replaced by summary sub-dicts.
    """
    if not per_seed_results:
        return {}

    all_keys = per_seed_results[0].keys()
    aggregated: dict = {}

    for key in all_keys:
        values = [r[key] for r in per_seed_results if key in r]
        numeric_values = extract_numeric_values(values)

        if len(numeric_values) == len(values) and len(numeric_values) > 0:
            aggregated[key] = build_seed_summary(numeric_values)
        else:
            aggregated[key] = values[0] if values else None

    return aggregated


def collect_key_values(per_seed_results: list[dict], key: str) -> list[float]:
    result = []

    for r in per_seed_results:
        try:
            fv = float(r.get(key, float("nan")))
            result.append(fv if math.isfinite(fv) else float("nan"))
        except (TypeError, ValueError):
            result.append(float("nan"))

    return result


def find_key_outliers(all_values: list[float], ids: list[int], key: str) -> list[tuple[int, str]]:
    finite_vals = [v for v in all_values if math.isfinite(v)]

    if len(finite_vals) < 3:
        return []

    mean = float(np.mean(finite_vals))
    std = float(np.std(finite_vals, ddof=1))

    if std < 1e-10:
        return []

    return [
        (seed_id, key)
        for seed_id, v in zip(ids, all_values)
        if math.isfinite(v) and abs(v - mean) > OUTLIER_SIGMA * std
    ]


def detect_outliers(
    per_seed_results: list[dict], seed_ids: list[int] | None = None
) -> list[tuple[int, str]]:
    """Return list of (seed_id, metric_name) pairs where the value deviates > 3σ.

    Parameters
    ----------
    per_seed_results:
        Per-seed result dicts.
    seed_ids:
        Optional list of seed ints parallel to per_seed_results.

    Returns
    -------
    List of (seed_id, metric_name) tuples for each outlier found.
    """
    if len(per_seed_results) < 3:
        return []

    ids = seed_ids if seed_ids is not None else list(range(len(per_seed_results)))
    outliers: list[tuple[int, str]] = []

    for key in per_seed_results[0]:
        values = collect_key_values(per_seed_results, key)
        outliers.extend(find_key_outliers(values, ids, key))

    return outliers


def format_numeric_row(key: str, val: dict) -> str:
    mean = val["mean"]
    std = val["std"]
    lo = val["ci95_low"]
    hi = val["ci95_high"]
    mean_str = f"{mean:.4f}" if math.isfinite(mean) else "nan"
    std_str = f"{std:.4f}" if math.isfinite(std) else "nan"
    ci_str = (
        f"[{lo:.4f}, {hi:.4f}]"
        if (math.isfinite(lo) and math.isfinite(hi))
        else "nan"
    )
    return f"  {key:<33} {mean_str:>12} {std_str:>10} {ci_str:>22}"


def format_stats_table(aggregated: dict) -> str:
    """Return a human-readable table string from an aggregated dict.

    Columns: metric | mean | std | 95% CI
    Only numeric entries (dicts with 'mean' key) are shown.
    """
    lines = [
        f"\n{'Metric':<35} {'Mean':>12} {'Std':>10} {'95% CI':>22}",
        "-" * 82,
    ]

    for key, val in aggregated.items():
        if isinstance(val, dict) and "mean" in val:
            lines.append(format_numeric_row(key, val))

    lines.append("")
    return "\n".join(lines)
