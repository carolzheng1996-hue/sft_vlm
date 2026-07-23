from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .signal_generator import render_seasonality, render_signal, sample_signal_spec


CORRELATION_BANDS = {
    "low": (0.10, 0.40),
    "medium": (0.40, 0.70),
    "high": (0.70, 0.90),
}

GENERAL_PATTERNS = {
    "multi_period", "missing_blocks", "covariates", "leakage",
    "residual_period", "residual_hetero",
}


def validate_relation_config(config: dict[str, Any] | None) -> dict[str, Any]:
    raw = config or {}
    mix = {str(key): float(value) for key, value in raw.get("general_correlation_mix", {"low": .25, "medium": .50, "high": .25}).items()}
    if set(mix) != set(CORRELATION_BANDS) or any(value < 0 for value in mix.values()) or abs(sum(mix.values()) - 1) > 1e-6:
        raise ValueError("general_correlation_mix must contain low/medium/high, be non-negative, and sum to 1")
    return {"quality_gate": bool(raw.get("quality_gate", True)), "max_attempts": int(raw.get("max_attempts", 12)), "general_correlation_mix": mix}


def sample_relation_profile(pattern: str, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(seed + 314_159)
    if pattern in GENERAL_PATTERNS:
        names = list(CORRELATION_BANDS)
        probabilities = [config["general_correlation_mix"][name] for name in names]
        tier = str(rng.choice(names, p=probabilities))
        band = CORRELATION_BANDS[tier]
        return {"type": "general", "tier": tier, "target_range": list(band), "target": float(rng.uniform(*band))}
    fixed = {
        "scaled_groups": ("scaled_shape", .75, .95),
        "phase_groups": ("phase_shift", -.10, .75),
        "distribution_mismatch": ("distribution_not_shape", -.40, .40),
        "motif": ("local_motif", .10, .65),
        "sequence_anomaly": ("sequence_anomaly", .60, .90),
        "heterogeneous_panel": ("heterogeneous", .05, .65),
        "hierarchical": ("hierarchical", .10, .65),
        "intermittent": ("intermittent", 0.0, .60),
    }
    kind, low, high = fixed.get(pattern, ("general", .10, .90))
    return {"type": kind, "tier": None, "target_range": [low, high], "target": float(rng.uniform(max(0, low), high))}


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return (values - float(np.mean(values))) / (float(np.std(values)) + 1e-8)


def _factor(spec: dict[str, Any], n: int, seed: int, *, seasonal_only: bool = False) -> np.ndarray:
    if seasonal_only and spec.get("seasonality"):
        values = render_seasonality(n, spec["seasonality"])
    else:
        values, _ = render_signal(np.random.default_rng(seed), n, spec, idiosyncratic_scale=.35)
    return _standardize(values)


def _weights(profile: dict[str, Any], same_group_pair_share: float) -> tuple[float, float, float]:
    kind = profile["type"]
    target = float(profile["target"])
    if kind == "hierarchical":
        common = float(np.clip(target, .18, .60)); group = float(min(.45, .85 - common)); individual = 1.0 - common - group
        return common, group, individual
    if kind == "scaled_shape":
        group = .02
    elif kind == "hierarchical":
        group = float(min(.45, max(.25, .68 - target)))
    elif kind in {"heterogeneous", "local_motif"}:
        group = .12
    elif kind in {"phase_shift", "distribution_not_shape", "sequence_anomaly"}:
        group = 0.0
    else:
        group = float(min(.18, max(.05, (1 - target) * .25)))
    common = float(np.clip(target - same_group_pair_share * group, .02, .92))
    individual = max(.04, 1.0 - common - group)
    total = common + group + individual
    return common / total, group / total, individual / total


def synthesize_panel(
    n: int,
    k: int,
    pattern: str,
    complexity: str,
    base_spec: dict[str, Any],
    seed: int,
    profile: dict[str, Any],
    anomalous_index: int | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(seed + 271_828)
    group_count = 3 if pattern == "phase_groups" else 2
    group_index = [index % group_count for index in range(k)]
    same_pairs = sum(group_index[i] == group_index[j] for i in range(k) for j in range(i + 1, k))
    pair_count = max(1, k * (k - 1) // 2)
    weights = _weights(profile, same_pairs / pair_count)

    common_seed = int(rng.integers(0, 2**31 - 1))
    common = _factor(base_spec, n, common_seed, seasonal_only=pattern == "phase_groups")
    group_specs: list[dict[str, Any]] = []
    group_factors: list[np.ndarray] = []
    group_seeds: list[int] = []
    for _ in range(group_count):
        factor_seed = int(rng.integers(0, 2**31 - 1)); group_seeds.append(factor_seed)
        spec_rng = np.random.default_rng(factor_seed + 17)
        spec = sample_signal_spec(spec_rng, n, complexity, allow_trend=True, seasonal_count=1)
        group_specs.append(spec); group_factors.append(_factor(spec, n, factor_seed))

    panels: list[np.ndarray] = []
    transformations: dict[str, Any] = {}
    individual_specs: dict[str, Any] = {}
    phase_lags: dict[str, int] = {}
    period = float(base_spec.get("seasonality", [{}])[0].get("period", max(6, n // 12))) if base_spec.get("seasonality") else max(6.0, n / 12)
    for index in range(k):
        series_id = f"s{index + 1:02d}"
        individual_seed = int(rng.integers(0, 2**31 - 1))
        spec_rng = np.random.default_rng(individual_seed + 29)
        individual_spec = sample_signal_spec(spec_rng, n, complexity, allow_trend=True, seasonal_count=1)
        individual = _factor(individual_spec, n, individual_seed)
        individual_specs[series_id] = {"seed": individual_seed, "spec": individual_spec}
        wc, wg, wi = weights
        common_for_series = common
        if pattern == "phase_groups":
            lag = int(round((group_index[index] / group_count) * period))
            phase_lags[series_id] = lag
            common_for_series = np.roll(common, lag)
            wc, wg, wi = .88, 0.0, .12
        if pattern == "distribution_mismatch":
            blocks = np.array_split(common, max(5, min(16, n // 35)))
            common_for_series = np.concatenate([blocks[i] for i in rng.permutation(len(blocks))])
            wc, wg, wi = .82, 0.0, .18
        if pattern == "sequence_anomaly" and index == anomalous_index:
            wc, wg, wi = .04, 0.0, .96
        combined = math.sqrt(wc) * common_for_series + math.sqrt(wg) * group_factors[group_index[index]] + math.sqrt(wi) * individual
        scale = float(rng.uniform(.82, 1.22)); offset = float(rng.uniform(-6, 7)); amplitude = float(rng.uniform(9, 16))
        if pattern == "scaled_groups":
            scale = float(.60 + .28 * index); offset = float(rng.uniform(-2, 2))
        elif pattern == "heterogeneous_panel":
            scale = float(rng.uniform(.55, 1.75)); offset = float(rng.uniform(-12, 16)); amplitude = float(rng.uniform(8, 19))
        panels.append(45.0 + offset + amplitude * scale * combined)
        transformations[series_id] = {"scale": scale, "offset": offset, "amplitude": amplitude, "group": f"g{group_index[index] + 1}", "weights": {"common": wc, "group": wg, "individual": wi}}

    truth = {
        "profile": profile,
        "common_latent_signal": True,
        "common_factor": {"seed": common_seed, "spec": base_spec},
        "group_assignment": {f"s{index + 1:02d}": f"g{group_index[index] + 1}" for index in range(k)},
        "group_factors": {f"g{index + 1}": {"seed": group_seeds[index], "spec": group_specs[index]} for index in range(group_count)},
        "individual_factors": individual_specs,
        "series_transformations": transformations,
        "base_variance_weights": {"common": weights[0], "group": weights[1], "individual": weights[2]},
        "phase_lags": phase_lags,
        "negative_transfer_group": [],
    }
    return panels, truth


def _off_diagonal(frame: pd.DataFrame) -> np.ndarray:
    values = frame.to_numpy(dtype=float)
    return values[np.triu_indices_from(values, 1)]


def _safe_stats(values: np.ndarray) -> dict[str, float | None]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return {"mean": None, "min": None, "max": None, "median": None}
    return {"mean": float(np.mean(finite)), "min": float(np.min(finite)), "max": float(np.max(finite)), "median": float(np.median(finite))}


def _best_lag_correlation(first: pd.Series, second: pd.Series, max_lag: int) -> float | None:
    best: float | None = None
    for lag in range(-max_lag, max_lag + 1):
        value = first.corr(second.shift(lag))
        if pd.notna(value): best = max(best if best is not None else -1.0, float(value))
    return best


def relation_metrics(frame: pd.DataFrame, pattern: str, truth: dict[str, Any]) -> dict[str, Any] | None:
    value_column = "value" if "value" in frame else "actual" if "actual" in frame else None
    if "series_id" in frame:
        if value_column is None or frame["series_id"].nunique() < 2:
            return None
        wide = frame.pivot_table(index="time", columns="series_id", values=value_column, aggfunc="mean").sort_index()
    else:
        primary = [column for column in frame.columns if re.fullmatch(r"s\d+", str(column))]
        actual = [column for column in frame.columns if re.fullmatch(r"s\d+__actual", str(column))]
        selected = primary or actual
        if len(selected) < 2 or "time" not in frame:
            return None
        value_column = "value" if primary else "actual"
        wide = frame[["time", *selected]].groupby("time", sort=False).mean(numeric_only=True).sort_index()
        wide.columns = [str(column).split("__", 1)[0] for column in wide.columns]
    raw_corr = wide.corr(min_periods=max(8, len(wide) // 10))
    diff_corr = wide.diff().corr(min_periods=max(8, len(wide) // 10))
    detrended = pd.DataFrame(index=wide.index)
    time = np.arange(len(wide), dtype=float)
    for column in wide:
        observed = wide[column].notna().to_numpy()
        if observed.sum() < 3:
            detrended[column] = np.nan
        else:
            fit = np.polyfit(time[observed], wide[column].to_numpy()[observed], 1)
            detrended[column] = wide[column] - np.polyval(fit, time)
    detrended_corr = detrended.corr(min_periods=max(8, len(wide) // 10))
    overlap = wide.notna().astype(int).T.dot(wide.notna().astype(int))
    raw_values = _off_diagonal(raw_corr)
    assignment = truth.get("generator", {}).get("components", {}).get("multi_series", {}).get("group_assignment", {})
    within: list[float] = []; between: list[float] = []
    for i, first in enumerate(raw_corr.columns):
        for second in raw_corr.columns[i + 1:]:
            value = raw_corr.loc[first, second]
            if pd.isna(value): continue
            (within if assignment.get(str(first)) == assignment.get(str(second)) else between).append(float(value))
    metrics: dict[str, Any] = {
        "value_column": value_column,
        "raw_pearson": _safe_stats(raw_values),
        "first_difference_pearson": _safe_stats(_off_diagonal(diff_corr)),
        "linear_detrended_pearson": _safe_stats(_off_diagonal(detrended_corr)),
        "within_group_mean": float(np.mean(within)) if within else None,
        "between_group_mean": float(np.mean(between)) if between else None,
        "overlap": _safe_stats(_off_diagonal(overlap)),
        "all_pairs_above_0_90": bool(len(raw_values) and np.all(raw_values[np.isfinite(raw_values)] > .90)),
    }
    if pattern == "phase_groups":
        columns = list(wide.columns); best = []
        for index in range(1, len(columns)):
            value = _best_lag_correlation(wide[columns[0]], wide[columns[index]], min(60, max(3, len(wide) // 6)))
            if value is not None: best.append(value)
        metrics["best_lag_correlation_mean"] = float(np.mean(best)) if best else None
    if pattern == "distribution_mismatch":
        columns = list(wide.columns); distances = []
        for index in range(1, len(columns)):
            first = wide[columns[0]].dropna(); second = wide[columns[index]].dropna()
            pooled = float(pd.concat([first, second]).std()) + 1e-8
            distances.append(float(wasserstein_distance(first, second) / pooled))
        metrics["normalized_wasserstein_mean"] = float(np.mean(distances)) if distances else None
    if pattern == "motif":
        motif_truth = truth.get("generator", {}).get("events", {}).get("motif", {})
        local = []
        reference_id = str(wide.columns[0]); reference_meta = motif_truth.get(reference_id)
        if reference_meta:
            start = reference_meta["starts"][0]; length = reference_meta["length"]
            reference = wide[reference_id].iloc[start:start + length].reset_index(drop=True)
            for column in wide.columns[1:]:
                meta = motif_truth.get(str(column))
                if meta:
                    segment = wide[column].iloc[meta["starts"][0]:meta["starts"][0] + length].reset_index(drop=True)
                    value = reference.corr(segment)
                    if pd.notna(value): local.append(float(value))
        metrics["motif_local_correlation_mean"] = float(np.mean(local)) if local else None
    if pattern == "sequence_anomaly":
        anomalous = truth.get("anomalous_series")
        normals = [column for column in wide.columns if str(column) != anomalous]
        normal_corr = wide[normals].corr() if len(normals) > 1 else pd.DataFrame()
        metrics["normal_group_mean"] = _safe_stats(_off_diagonal(normal_corr))["mean"] if len(normals) > 1 else None
        metrics["anomaly_to_normal_mean"] = float(np.nanmean([raw_corr.loc[anomalous, column] for column in normals])) if anomalous in raw_corr else None
    return metrics


def relation_passes(pattern: str, profile: dict[str, Any], metrics: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if metrics is None:
        return True, []
    reasons: list[str] = []
    mean = metrics["raw_pearson"]["mean"]
    if mean is None:
        return False, ["raw_correlation_unavailable"]
    kind = profile["type"]
    low, high = profile["target_range"]
    if kind == "general":
        if not low - .06 <= mean <= high + .06: reasons.append("general_correlation_out_of_band")
        if mean > .90: reasons.append("general_mean_above_0_90")
        if metrics["all_pairs_above_0_90"]: reasons.append("all_pairs_above_0_90")
    elif kind == "scaled_shape" and not .75 <= mean <= .95:
        reasons.append("scaled_shape_correlation_out_of_band")
    elif kind == "phase_shift":
        if not -.10 <= mean <= .75: reasons.append("phase_zero_lag_out_of_band")
        if (metrics.get("best_lag_correlation_mean") or -1) < .80: reasons.append("phase_best_lag_too_low")
    elif kind == "distribution_not_shape":
        if abs(mean) > .40: reasons.append("distribution_shape_correlation_too_high")
        if (metrics.get("normalized_wasserstein_mean") or 99) > .45: reasons.append("distribution_distance_too_high")
    elif kind == "local_motif":
        if not .10 <= mean <= .65: reasons.append("motif_global_correlation_out_of_band")
        if (metrics.get("motif_local_correlation_mean") or -1) < .80: reasons.append("motif_local_similarity_too_low")
    elif kind == "sequence_anomaly":
        normal = metrics.get("normal_group_mean"); anomaly = metrics.get("anomaly_to_normal_mean")
        if normal is None or not .60 <= normal <= .90: reasons.append("normal_group_correlation_out_of_band")
        if anomaly is None or anomaly > .45: reasons.append("anomaly_not_separated")
    elif kind == "heterogeneous" and not .05 <= mean <= .65:
        reasons.append("heterogeneous_correlation_out_of_band")
    elif kind == "hierarchical":
        within = metrics.get("within_group_mean"); between = metrics.get("between_group_mean")
        if within is None or not .55 <= within <= .90: reasons.append("hierarchy_within_group_out_of_band")
        if between is None or not .10 <= between <= .65: reasons.append("hierarchy_between_group_out_of_band")
        if within is None or between is None or within - between < .15: reasons.append("hierarchy_group_gap_too_small")
    elif kind == "intermittent" and not 0 <= mean <= .60:
        reasons.append("intermittent_correlation_out_of_band")
    return not reasons, reasons
