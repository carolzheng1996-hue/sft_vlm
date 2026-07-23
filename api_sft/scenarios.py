from __future__ import annotations

import math
import os
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/api_sft_matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/api_sft_cache")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf

from .common import iter_jsonl, read_json, sha256_file, stable_hash, write_json, write_jsonl
from .multiseries import relation_metrics, relation_passes, sample_relation_profile, synthesize_panel, validate_relation_config
from .signal_generator import (
    DEFAULT_COMPLEXITY_MIX,
    GENERATOR_VERSION,
    complexity_schedule,
    legacy_trend_slope,
    render_signal,
    sample_signal_spec,
    validate_complexity_mix,
)


WIDE_PANEL_LAYOUT = "wide_panel_v1"
TRAINING_CURVE_LAYOUT = "training_curve_v1"
_WIDE_SERIES_RE = re.compile(r"^(s\d+)(?:__(.+))?$")


ARCHETYPES: list[dict[str, Any]] = [
    {"name":"stationary_short_schema","task":"tool_use","goal":"forecast","count":"single","length":"short","pattern":"stationary","mode":"text_only","image_value":"low"},
    {"name":"trend_period_profile","task":"data_profile","goal":"forecast","count":"single","length":"long","pattern":"trend_period","mode":"image_text","image_value":"high"},
    {"name":"multi_period_profile","task":"data_profile","goal":"diagnosis","count":"multiple","length":"long","pattern":"multi_period","mode":"image_text","image_value":"high"},
    {"name":"missingness_profile","task":"data_profile","goal":"diagnosis","count":"multiple","length":"long","pattern":"missing_blocks","mode":"image_text","image_value":"high"},
    {"name":"intermittent_profile","task":"data_profile","goal":"forecast","count":"single","length":"long","pattern":"intermittent","mode":"text_then_image","image_value":"medium"},
    {"name":"short_scaled_similarity","task":"similarity_analysis","goal":"forecast","count":"multiple","length":"short","pattern":"scaled_groups","mode":"text_then_image","image_value":"medium"},
    {"name":"long_shifted_similarity","task":"similarity_analysis","goal":"sequence_anomaly_detection","count":"multiple","length":"long","pattern":"phase_groups","mode":"image_text","image_value":"high"},
    {"name":"distribution_shape_mismatch","task":"similarity_analysis","goal":"diagnosis","count":"multiple","length":"long","pattern":"distribution_mismatch","mode":"image_text","image_value":"high"},
    {"name":"motif_similarity","task":"similarity_analysis","goal":"sequence_anomaly_detection","count":"multiple","length":"long","pattern":"motif","mode":"image_text","image_value":"high"},
    {"name":"anomalous_series_similarity","task":"similarity_analysis","goal":"sequence_anomaly_detection","count":"multiple","length":"long","pattern":"sequence_anomaly","mode":"image_text","image_value":"high"},
    {"name":"point_anomaly_changepoint","task":"data_profile","goal":"point_anomaly_detection","count":"single","length":"long","pattern":"anomaly_change","mode":"image_text","image_value":"high"},
    {"name":"missingness_panel","task":"tool_use","goal":"diagnosis","count":"multiple","length":"long","pattern":"missing_blocks","mode":"image_text","image_value":"high"},
    {"name":"short_model_selection","task":"model_selection","goal":"forecast","count":"single","length":"short","pattern":"short_trend","mode":"text_then_image","image_value":"medium"},
    {"name":"panel_model_selection","task":"model_selection","goal":"forecast","count":"multiple","length":"long","pattern":"heterogeneous_panel","mode":"text_then_image","image_value":"high"},
    {"name":"covariate_model_selection","task":"model_selection","goal":"forecast","count":"multiple","length":"long","pattern":"covariates","mode":"text_then_image","image_value":"medium"},
    {"name":"intermittent_model_selection","task":"model_selection","goal":"forecast","count":"multiple","length":"long","pattern":"intermittent","mode":"text_then_image","image_value":"medium"},
    {"name":"hierarchical_model_selection","task":"model_selection","goal":"forecast","count":"multiple","length":"long","pattern":"hierarchical","mode":"text_then_image","image_value":"medium"},
    {"name":"interval_model_selection","task":"model_selection","goal":"forecast","count":"single","length":"long","pattern":"interval","mode":"text_then_image","image_value":"medium"},
    {"name":"residual_autocorrelation","task":"model_result_analysis","goal":"diagnosis","count":"single","length":"long","pattern":"residual_period","mode":"image_text","image_value":"high"},
    {"name":"residual_heteroscedastic","task":"model_result_analysis","goal":"diagnosis","count":"multiple","length":"long","pattern":"residual_hetero","mode":"image_text","image_value":"high"},
    {"name":"training_overfit","task":"model_result_analysis","goal":"monitoring_retraining","count":"single","length":"short","pattern":"overfit","mode":"image_text","image_value":"high"},
    {"name":"training_underfit","task":"model_result_analysis","goal":"monitoring_retraining","count":"single","length":"short","pattern":"underfit","mode":"text_then_image","image_value":"medium"},
    {"name":"online_drift","task":"model_result_analysis","goal":"monitoring_retraining","count":"multiple","length":"long","pattern":"drift","mode":"image_text","image_value":"high"},
    {"name":"leakage_metric_rules","task":"tool_use","goal":"forecast","count":"multiple","length":"short","pattern":"leakage","mode":"text_only","image_value":"low"},
    {"name":"anomaly_tool_route","task":"tool_use","goal":"point_anomaly_detection","count":"single","length":"long","pattern":"anomaly_change","mode":"image_text","image_value":"high"},
    {"name":"period_tool_route","task":"tool_use","goal":"diagnosis","count":"single","length":"long","pattern":"trend_period","mode":"text_then_image","image_value":"medium"},
    {"name":"similarity_tool_route","task":"tool_use","goal":"diagnosis","count":"multiple","length":"long","pattern":"scaled_groups","mode":"text_then_image","image_value":"medium"},
    {"name":"conflicting_tool_route","task":"tool_use","goal":"diagnosis","count":"single","length":"long","pattern":"multi_period","mode":"text_then_image","image_value":"medium"},
    {"name":"sequence_anomaly_group","task":"model_selection","goal":"sequence_anomaly_detection","count":"multiple","length":"long","pattern":"sequence_anomaly","mode":"image_text","image_value":"high"},
]

TASK_RATIOS = {"data_profile": .25, "similarity_analysis": .18, "model_result_analysis": .20, "model_selection": .22, "tool_use": .15}


@dataclass
class Scenario:
    manifest: dict[str, Any]
    frame: pd.DataFrame
    truth: dict[str, Any]


def _wide_name(alias: str, source_column: str, primary_metric: str | None) -> str:
    return alias if source_column == primary_metric else f"{alias}__{source_column}"


def to_wide_panel(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert an observed long panel to wide_panel_v1 without aggregation.

    The first observation for each (time, series) is placed on the shared base
    row. Every additional duplicate observation becomes its own sparse repeated
    time row. The internal occurrence rank is deliberately not exposed.
    """

    if "series_id" not in frame.columns:
        return frame.copy(), {
            "version": TRAINING_CURVE_LAYOUT,
            "time_column": "step" if "step" in frame.columns else str(frame.columns[0]),
            "columns": {},
            "duplicate_policy": "not_applicable",
            "source_row_count": len(frame),
            "wide_row_count": len(frame),
        }
    if "time" not in frame.columns:
        raise ValueError("Long panel requires time and series_id columns")
    source = frame.reset_index(drop=True).copy()
    source["__source_order"] = np.arange(len(source))
    source["__occurrence"] = source.groupby(["time", "series_id"], sort=False, dropna=False).cumcount()
    source_series = list(dict.fromkeys(str(value) for value in source["series_id"].tolist()))
    aliases = {series_id: f"s{index + 1:02d}" for index, series_id in enumerate(source_series)}
    metric_columns = [column for column in frame.columns if column not in {"time", "series_id"}]
    primary_metric = "value" if "value" in metric_columns else "actual" if "actual" in metric_columns else None
    column_map: dict[str, dict[str, Any]] = {}
    ordered_columns: list[str] = ["time"]
    if primary_metric:
        for source_series_id in source_series:
            name = _wide_name(aliases[source_series_id], primary_metric, primary_metric)
            ordered_columns.append(name)
            column_map[name] = {"source_series_id": source_series_id, "source_column": primary_metric, "role": "primary"}
    for source_series_id in source_series:
        for metric in metric_columns:
            if metric == primary_metric:
                continue
            name = _wide_name(aliases[source_series_id], metric, primary_metric)
            ordered_columns.append(name)
            column_map[name] = {"source_series_id": source_series_id, "source_column": metric, "role": "metric"}

    rows: list[dict[str, Any]] = []
    for time_value in source["time"].drop_duplicates().tolist():
        at_time = source[source["time"].eq(time_value)]
        base: dict[str, Any] = {"time": time_value}
        for _, item in at_time[at_time["__occurrence"].eq(0)].iterrows():
            source_series_id = str(item["series_id"])
            for metric in metric_columns:
                base[_wide_name(aliases[source_series_id], metric, primary_metric)] = item[metric]
        rows.append(base)
        for _, item in at_time[at_time["__occurrence"].gt(0)].sort_values("__source_order").iterrows():
            duplicate: dict[str, Any] = {"time": time_value}
            source_series_id = str(item["series_id"])
            for metric in metric_columns:
                duplicate[_wide_name(aliases[source_series_id], metric, primary_metric)] = item[metric]
            rows.append(duplicate)
    wide = pd.DataFrame(rows).reindex(columns=ordered_columns)
    source_duplicates = int(frame.duplicated(["time", "series_id"]).sum())
    wide_duplicates = int(wide.duplicated(["time"]).sum())
    if source_duplicates != wide_duplicates:
        raise RuntimeError(
            f"Duplicate preservation failed: source={source_duplicates}, wide={wide_duplicates}"
        )
    mapping = {
        "version": WIDE_PANEL_LAYOUT,
        "time_column": "time",
        "series": [
            {"source_series_id": source_series_id, "alias": aliases[source_series_id]}
            for source_series_id in source_series
        ],
        "primary_metric": primary_metric,
        "columns": column_map,
        "duplicate_policy": "sparse_repeated_time_rows_no_aggregation",
        "source_row_count": len(frame),
        "wide_row_count": len(wide),
        "source_duplicate_time_series_rows": source_duplicates,
        "wide_duplicate_time_rows": wide_duplicates,
    }
    return wide, mapping


def _metric_wide_columns(frame: pd.DataFrame, metric: str) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        match = _WIDE_SERIES_RE.fullmatch(str(column))
        if not match:
            continue
        observed_metric = match.group(2) or "value"
        if observed_metric == metric:
            result.append(str(column))
    return result


def _source_columns_for_metric(frame: pd.DataFrame, metric: str) -> list[str]:
    if metric in frame.columns:
        return [metric]
    return _metric_wide_columns(frame, metric)


def wide_to_long_panel(frame: pd.DataFrame, mapping: dict[str, Any]) -> pd.DataFrame:
    """Derive a canonical long analysis view from wide_panel_v1 input.

    This helper is for internal quality and plotting calculations only. It does
    not write an occurrence column or alter the persisted V4 dataset.
    """

    if mapping.get("version") != WIDE_PANEL_LAYOUT:
        return frame.copy()
    if "time" not in frame.columns:
        raise ValueError("wide_panel_v1 requires a time column")
    base_rows = ~frame["time"].duplicated(keep="first")
    pieces: list[pd.DataFrame] = []
    columns_map = mapping.get("columns") or {}
    for series in mapping.get("series") or []:
        source_series_id = str(series["source_series_id"])
        aliases = [name for name, item in columns_map.items() if str(item.get("source_series_id")) == source_series_id]
        if not aliases:
            continue
        observed_duplicate = frame[aliases].notna().any(axis=1)
        selected = frame.loc[base_rows | observed_duplicate, ["time", *aliases]].copy()
        selected.insert(1, "series_id", source_series_id)
        selected = selected.rename(columns={name: columns_map[name]["source_column"] for name in aliases})
        pieces.append(selected)
    if not pieces:
        return pd.DataFrame(columns=["time", "series_id"])
    return pd.concat(pieces, ignore_index=True, sort=False)


def _length(rng: np.random.Generator, label: str) -> int:
    return int(rng.integers(48, 121)) if label == "short" else int(rng.integers(720, 1101))


def _base_options(pattern: str, complexity: str) -> tuple[bool, int | None, str | None]:
    if pattern == "stationary":
        return False, 0, "gaussian" if complexity == "controlled" else "ar1"
    if pattern == "short_trend":
        return True, 0, None
    if pattern == "multi_period":
        return True, 2 if complexity == "controlled" else 3, None
    if pattern in {"trend_period", "phase_groups", "scaled_groups"}:
        return True, 1 if complexity == "controlled" else 2, None
    return True, None, None


def _training_frame(rng: np.random.Generator, n: int, pattern: str, complexity: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    epoch = np.arange(1, n + 1)
    oscillation = np.zeros(n)
    if complexity == "confounded":
        oscillation = 0.04 * np.sin(2 * np.pi * epoch / max(5, n // 9))
    if pattern == "overfit":
        onset = int(n * rng.uniform(.48, .68))
        train = 1.45 * np.exp(-epoch / max(12, n / 4)) + .10 + rng.normal(0, .015, n)
        valid = 1.30 * np.exp(-epoch / max(10, n / 5)) + .18 + np.maximum(0, epoch - onset) * rng.uniform(.009, .016) + oscillation + rng.normal(0, .025, n)
        issue = "overfit"
    else:
        onset = None
        train = 1.20 * np.exp(-epoch / max(65, n)) + .58 + oscillation + rng.normal(0, .025, n)
        valid = 1.30 * np.exp(-epoch / max(70, n)) + .65 + 1.2 * oscillation + rng.normal(0, .03, n)
        issue = "underfit_with_oscillation" if complexity == "confounded" else "underfit"
    return pd.DataFrame({"step": epoch, "train_loss": train, "validation_loss": valid}), {"training_issue": issue, "onset": onset, "oscillation": bool(np.any(oscillation))}


def _drift_frame(rng: np.random.Generator, n: int, k: int, complexity: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = int(n * rng.uniform(.55, .72))
    drift_type = "combined" if complexity == "controlled" else str(rng.choice(["data_drift", "concept_drift", "pipeline_drift", "combined"]))
    rows: list[dict[str, Any]] = []
    affected = set(rng.choice(k, size=max(1, k // 2), replace=False).tolist()) if complexity == "confounded" else set(range(k))
    for series_index in range(k):
        error = np.abs(rng.normal(2.0 + .15 * series_index, .8, n))
        missing = np.clip(rng.normal(.01, .004, n), 0, 1)
        psi = np.clip(rng.normal(.05, .012, n), 0, 1)
        if series_index in affected:
            ramp = np.linspace(0, 1, n - start)
            if drift_type in {"concept_drift", "combined"}:
                error[start:] += 9.0 * ramp
            if drift_type in {"data_drift", "combined"}:
                psi[start:] += .32 * ramp
                error[start:] += 3.0 * ramp
            if drift_type in {"pipeline_drift", "combined"}:
                missing[start:] += .13 * ramp
                error[start:] += 4.5 * ramp
        for time in range(n):
            rows.append({"time": time, "series_id": f"s{series_index + 1:02d}", "absolute_error": error[time], "missing_rate": missing[time], "feature_psi": psi[time]})
    return pd.DataFrame(rows), {"drift_start": start, "drift_type": drift_type, "affected_series": [f"s{i + 1:02d}" for i in sorted(affected)], "retrain_should_trigger": True}


def _intermittent_series(rng: np.random.Generator, n: int, complexity: str, scale: float, shared_events: np.ndarray | None = None, synchronization: float = 0.0) -> tuple[np.ndarray, dict[str, Any]]:
    base_probability = float(rng.uniform(.07, .19))
    probability = np.full(n, base_probability)
    if complexity in {"compositional", "confounded"}:
        probability *= 1 + .45 * np.sin(2 * np.pi * np.arange(n) / max(7, int(n / 8)))
    probability = np.clip(probability, .01, .65)
    individual_events = rng.random(n) < probability
    if shared_events is not None and synchronization > 0:
        use_shared = rng.random(n) < synchronization
        events = np.where(use_shared, shared_events, individual_events)
    else:
        events = individual_events
    amounts = rng.lognormal(mean=3.0, sigma=.75 if complexity != "confounded" else 1.05, size=n) * scale
    return np.where(events, amounts, 0.0), {"base_event_probability": base_probability, "zero_ratio": float(np.mean(~events)), "long_tail": True, "event_synchronization": synchronization}


def _inject_anomaly_and_change(rng: np.random.Generator, y: np.ndarray, complexity: str) -> tuple[np.ndarray, dict[str, Any]]:
    result = y.copy()
    n = len(result)
    change_point = int(n * rng.uniform(.50, .70))
    shift = float(rng.choice([-1, 1]) * rng.uniform(7, 16))
    result[change_point:] += shift
    change_kind = "mean"
    if complexity in {"compositional", "confounded"}:
        multiplier = float(rng.uniform(1.35, 2.4))
        tail_mean = float(np.nanmean(result[change_point:]))
        result[change_point:] = tail_mean + multiplier * (result[change_point:] - tail_mean)
        change_kind = "mean_and_variance"
    point_indices = sorted(rng.choice(np.arange(max(2, int(n * .12)), min(n - 2, int(n * .88))), size=2, replace=False).astype(int).tolist())
    magnitudes = rng.choice([-1, 1], size=2) * rng.uniform(16, 30, size=2)
    result[point_indices] += magnitudes
    cluster: list[int] = []
    contextual: int | None = None
    if complexity != "controlled":
        cluster_start = int(n * rng.uniform(.24, .78))
        cluster = list(range(cluster_start, min(n, cluster_start + max(3, n // 80))))
        result[cluster] += float(rng.choice([-1, 1]) * rng.uniform(6, 12))
    if complexity == "confounded":
        contextual = int(np.argmax(result[:max(3, int(n * .8))]))
        result[contextual] *= float(rng.uniform(.35, .65))
    return result, {"change_point": change_point, "change_type": change_kind, "mean_shift": shift, "point_anomalies": point_indices, "anomaly_cluster": cluster, "contextual_anomaly": contextual}


def _inject_missingness(rng: np.random.Generator, y: np.ndarray, complexity: str, series_index: int, shared_start: int | None) -> tuple[np.ndarray, dict[str, Any]]:
    result = y.copy()
    n = len(result)
    block_start = shared_start if shared_start is not None else int(n * rng.uniform(.20, .72))
    block_length = max(2, int(n * rng.uniform(.045, .10)))
    block_end = min(n, block_start + block_length)
    result[block_start:block_end] = np.nan
    random_indices: list[int] = []
    conditional_indices: list[int] = []
    if complexity in {"compositional", "confounded"}:
        candidates = np.setdiff1d(np.arange(n), np.arange(block_start, block_end))
        size = max(1, int(n * rng.uniform(.01, .035)))
        random_indices = sorted(rng.choice(candidates, size=min(size, len(candidates)), replace=False).astype(int).tolist())
        result[random_indices] = np.nan
    if complexity == "confounded":
        finite = np.flatnonzero(np.isfinite(result))
        if len(finite):
            threshold = np.nanquantile(result, .86)
            candidates = finite[result[finite] >= threshold]
            conditional_indices = sorted(candidates[::max(1, len(candidates) // max(1, n // 100))].astype(int).tolist()[:max(1, n // 100)])
            result[conditional_indices] = np.nan
    mechanism = "synchronous_block" if shared_start is not None else "series_specific_block"
    if random_indices:
        mechanism += "+random"
    if conditional_indices:
        mechanism += "+value_dependent"
    return result, {"series": f"s{series_index + 1:02d}", "mechanism": mechanism, "block": [block_start, block_end - 1], "random_indices": random_indices, "conditional_indices": conditional_indices}


def _apply_time_index_issue(frame: pd.DataFrame, rng: np.random.Generator, pattern: str, complexity: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if pattern not in {"stationary", "leakage"}:
        return frame, {"type": "regular"}
    issue = "duplicate" if complexity == "controlled" else str(rng.choice(["duplicate", "gap", "unsorted"]))
    result = frame.copy()
    if issue == "duplicate":
        for _, indices in result.groupby("series_id").groups.items():
            ordered = list(indices)
            if len(ordered) > 4:
                result.loc[ordered[len(ordered) // 2], "time"] = result.loc[ordered[len(ordered) // 2 - 1], "time"]
    elif issue == "gap":
        midpoint = int(result["time"].max() * .55)
        result.loc[result["time"] >= midpoint, "time"] += int(rng.integers(2, 6))
    else:
        result = result.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1))).reset_index(drop=True)
    return result, {"type": issue}


def _add_covariates(frame: pd.DataFrame, rng: np.random.Generator, truth: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    time = result["time"].to_numpy(dtype=float)
    result["promo_flag"] = (np.mod(time, 30) < 4).astype(int)
    result["temperature"] = np.round(20 + 8 * np.sin(2 * np.pi * time / 365), 3)
    result["actual_delivery_delay"] = np.round(rng.normal(2, 1, len(result)), 3)
    if "value" in result:
        promo_effect = float(rng.uniform(2.0, 7.0))
        temperature_effect = float(rng.uniform(-.20, .25))
        lag = int(rng.choice([1, 2, 3]))
        lagged_promo = result.groupby("series_id")["promo_flag"].shift(lag).fillna(0)
        result["value"] += promo_effect * lagged_promo + temperature_effect * np.maximum(0, result["temperature"] - 20) ** 1.25
        truth["generator"]["covariates"] = {"promo_effect": promo_effect, "promo_lag": lag, "temperature_effect": temperature_effect, "unknown_at_prediction": ["actual_delivery_delay"]}
    truth["leakage_risk"] = ["actual_delivery_delay"]
    return result


def _add_leakage_columns(frame: pd.DataFrame, truth: dict[str, Any]) -> pd.DataFrame:
    result = frame.copy()
    result["target_t_plus_1"] = result.groupby("series_id")["value"].shift(-1)
    result["future_window_target_mean"] = result.groupby("series_id")["value"].transform(lambda values: values.iloc[::-1].rolling(4, min_periods=1).mean().iloc[::-1])
    truth.update({"leakage_columns": ["target_t_plus_1", "future_window_target_mean"], "image_needed": False})
    return result


def _create_scenario_once(index: int, archetype: dict[str, Any], seed: int, complexity: str, relation_profile: dict[str, Any] | None, generation_attempt: int) -> Scenario:
    attempt_seed = seed + generation_attempt * 1_000_003
    rng = np.random.default_rng(attempt_seed)
    n = _length(rng, archetype["length"])
    multi = archetype["count"] == "multiple"
    k = int(rng.integers(5, 9)) if multi else 1
    pattern = archetype["pattern"]
    generator_truth: dict[str, Any] = {"version": GENERATOR_VERSION, "complexity": complexity, "seed": seed, "attempt_seed": attempt_seed, "components": {}, "series_transformations": {}, "events": {}}
    truth: dict[str, Any] = {"pattern": pattern, "base_period": None, "trend_slope": 0.0, "series_count": k, "history_length": n, "generator": generator_truth}

    if pattern in {"overfit", "underfit"}:
        frame, training_truth = _training_frame(rng, n, pattern, complexity)
        generator_truth["components"]["training"] = training_truth
        truth.update(training_truth)
    elif pattern == "drift":
        frame, drift_truth = _drift_frame(rng, n, k, complexity)
        generator_truth["events"]["drift"] = drift_truth
        truth.update(drift_truth)
    else:
        allow_trend, seasonal_count, forced_noise = _base_options(pattern, complexity)
        signal_spec = sample_signal_spec(rng, n, complexity, allow_trend=allow_trend, seasonal_count=seasonal_count, forced_noise=forced_noise)
        generator_truth["components"]["signal"] = signal_spec
        truth["base_period"] = float(signal_spec["seasonality"][0]["period"]) if signal_spec["seasonality"] else None
        truth["seasonal_periods"] = [float(item["period"]) for item in signal_spec["seasonality"]]
        truth["trend_slope"] = legacy_trend_slope(n, signal_spec["trend"], seed)
        rows: list[dict[str, Any]] = []
        reference_y: np.ndarray | None = None
        anomalous_index = int(rng.integers(0, k)) if pattern == "sequence_anomaly" else None
        panel_values: list[np.ndarray] | None = None
        if multi and pattern != "intermittent":
            panel_values, multi_truth = synthesize_panel(n, k, pattern, complexity, signal_spec, attempt_seed, relation_profile or {"type":"general","tier":"medium","target_range":[.4,.7],"target":.55}, anomalous_index)
            generator_truth["components"]["multi_series"] = multi_truth
            generator_truth["series_transformations"] = dict(multi_truth["series_transformations"])
        elif multi:
            group_assignment = {f"s{series_index + 1:02d}": f"g{series_index % 2 + 1}" for series_index in range(k)}
            generator_truth["components"]["multi_series"] = {"profile":relation_profile,"common_latent_signal":False,"group_assignment":group_assignment,"group_factors":{},"individual_factors":{},"series_transformations":{},"event_synchrony":"independent","negative_transfer_group":[]}
        shared_missing_start = int(n * rng.uniform(.28, .62)) if pattern == "missing_blocks" and multi and complexity == "confounded" else None
        intermittent_groups: dict[str, np.ndarray] = {}
        intermittent_sync = 0.0
        if pattern == "intermittent" and multi:
            intermittent_sync = float(np.clip((relation_profile or {}).get("target", .2), .05, .60))
            base_probability = float(rng.uniform(.08, .18))
            for group in ["g1","g2"]: intermittent_groups[group] = rng.random(n) < base_probability
            generator_truth["components"]["multi_series"]["event_synchrony"] = {"mode":"group_mixture","probability":intermittent_sync}
        anomaly_truth: dict[str, Any] = {}
        missing_truth: dict[str, Any] = {}
        motif_truth: dict[str, Any] = {}
        for series_index in range(k):
            series_id = f"s{series_index + 1:02d}"
            scale = 1.0; offset = 0.0; phase_offset = 0.0; idiosyncratic_scale = 1.0
            if panel_values is None:
                scale = float(rng.uniform(.82,1.22)) if multi else 1.0
                offset = float(rng.uniform(-6,7)) if multi else 0.0
                idiosyncratic_scale = float(rng.uniform(.8,1.35)) if multi else 1.0
                generator_truth["series_transformations"][series_id] = {"scale":scale,"offset":offset,"phase_offset":phase_offset,"noise_scale":idiosyncratic_scale}
                if multi: generator_truth["components"]["multi_series"]["series_transformations"][series_id] = generator_truth["series_transformations"][series_id]
            if pattern == "intermittent":
                group = generator_truth["components"].get("multi_series", {}).get("group_assignment", {}).get(series_id)
                y, intermittent_truth = _intermittent_series(rng, n, complexity, scale, intermittent_groups.get(group), intermittent_sync)
                y = offset + y
                generator_truth["series_transformations"][series_id]["intermittent"] = intermittent_truth
            else:
                if panel_values is not None:
                    y = panel_values[series_index].copy()
                else:
                    y, _ = render_signal(rng, n, signal_spec, scale=scale, offset=offset, phase_offset=phase_offset, idiosyncratic_scale=idiosyncratic_scale)
            if pattern == "distribution_mismatch":
                if panel_values is None:
                    if reference_y is None:
                        reference_y = y.copy()
                    elif series_index == 1:
                        y = rng.permutation(reference_y)
                truth.update({"same_distribution_different_order": True, "dynamic_similarity": False})
            if pattern == "motif":
                motif_length = max(12, min(36, n // 12))
                # The motif must dominate its short local window without making
                # the complete histories globally similar.
                motif = np.sin(np.linspace(0, 2 * np.pi, motif_length)) * rng.uniform(28, 40)
                starts = [int(n * .18) + series_index * 2, int(n * .62) - series_index]
                for start in starts:
                    if start + motif_length <= n:
                        y[start:start + motif_length] += motif + rng.normal(0, .35, motif_length)
                motif_truth[f"s{series_index + 1:02d}"] = {"starts": starts, "length": motif_length}
            if pattern in {"anomaly_change", "sequence_anomaly"} and (pattern == "anomaly_change" or series_index == anomalous_index):
                y, injected = _inject_anomaly_and_change(rng, y, complexity)
                anomaly_truth[f"s{series_index + 1:02d}"] = injected
                if pattern == "anomaly_change" and k == 1:
                    truth["change_point"] = injected["change_point"]
                    truth["anomaly_indices"] = injected["point_anomalies"]
            if pattern == "sequence_anomaly" and series_index == anomalous_index:
                y += rng.uniform(5, 12) * np.sin(2 * np.pi * np.arange(n) / max(4, n / rng.uniform(5, 11)))
                truth["anomalous_series"] = f"s{series_index + 1:02d}"
                generator_truth["components"]["multi_series"]["negative_transfer_group"].append(series_id)
            if pattern == "heterogeneous_panel" and complexity == "confounded" and series_index >= max(1, k - 2):
                center = int(n * .62); width = max(8, n // 16)
                local_reversal = -rng.uniform(7, 13) * np.exp(-.5 * ((np.arange(n) - center) / width) ** 2)
                y += local_reversal
                generator_truth["components"]["multi_series"]["negative_transfer_group"].append(series_id)
                generator_truth["series_transformations"][series_id]["local_negative_transfer"] = {"center": center, "width": width, "amplitude": float(local_reversal.min())}
            if pattern == "missing_blocks":
                y, injected_missing = _inject_missingness(rng, y, complexity, series_index, shared_missing_start)
                missing_truth[f"s{series_index + 1:02d}"] = injected_missing
            actual = y.copy()
            if pattern in {"residual_period", "residual_hetero"}:
                residual_period = truth["base_period"] or max(5.0, n / 12)
                amplitude = np.linspace(1.0, 6.5, n) if pattern == "residual_hetero" else np.full(n, 3.0)
                innovations = rng.normal(0, .8, n)
                phi = .58 if complexity != "controlled" else 0.0
                for time in range(1, n):
                    innovations[time] += phi * innovations[time - 1]
                residual = amplitude * np.sin(2 * np.pi * np.arange(n) / residual_period) + innovations + rng.uniform(-.8, .8)
                if pattern == "residual_hetero":
                    tail_indices = rng.choice(n, size=max(2, n // 90), replace=False)
                    residual[tail_indices] *= rng.uniform(2.5, 4.0)
                prediction = actual - residual
                model_b = actual - rng.normal(.4, 2.4, n)
                width = np.nanquantile(np.abs(residual), .72)
                for time in range(n):
                    rows.append({"time": time, "series_id": f"s{series_index + 1:02d}", "horizon": time % min(24, max(2, n // 4)) + 1, "actual": actual[time], "prediction": prediction[time], "model_b_prediction": model_b[time], "lower_90": prediction[time] - width, "upper_90": prediction[time] + width, "residual": residual[time]})
                truth.update({"residual_autocorrelation_period": residual_period, "heteroscedastic": pattern == "residual_hetero", "systematic_bias": True})
                continue
            for time, value in enumerate(y):
                rows.append({"time": time, "series_id": f"s{series_index + 1:02d}", "value": value})
        frame = pd.DataFrame(rows)
        if anomaly_truth:
            generator_truth["events"]["anomaly_and_change"] = anomaly_truth
        if missing_truth:
            generator_truth["events"]["missingness"] = missing_truth
            truth["missing_blocks"] = {series: value["block"] for series, value in missing_truth.items()}
        if motif_truth:
            generator_truth["events"]["motif"] = motif_truth
            truth["motif_starts"] = {series: value["starts"] for series, value in motif_truth.items()}
        if "lower_90" in frame:
            truth["interval_coverage_90"] = float(((frame.actual >= frame.lower_90) & (frame.actual <= frame.upper_90)).mean())
        if pattern in {"scaled_groups", "phase_groups"}:
            truth.update({"similarity": "common_shape_with_scale_or_phase", "normalization_required": True, "dtw_useful": pattern == "phase_groups"})

        if pattern == "covariates":
            frame = _add_covariates(frame, rng, truth)
        if pattern == "hierarchical":
            ids = frame["series_id"].str.extract(r"(\d+)")[0].astype(int)
            frame["region"] = np.where(ids % 2 == 0, "north", "south")
            frame["store"] = "store_" + ((ids - 1) // 2 + 1).astype(str)
            truth["requires_reconciliation"] = True
            generator_truth["components"]["hierarchy"] = "region->store->series_id"
        if pattern == "leakage":
            frame = _add_leakage_columns(frame, truth)
        frame, time_issue = _apply_time_index_issue(frame, rng, pattern, complexity)
        generator_truth["events"]["time_index"] = time_issue

    if "series_id" in frame.columns:
        frame, wide_column_map = to_wide_panel(frame)
        data_layout = WIDE_PANEL_LAYOUT
    else:
        wide_column_map = {
            "version": TRAINING_CURVE_LAYOUT,
            "time_column": "step" if "step" in frame.columns else str(frame.columns[0]),
            "columns": {},
            "duplicate_policy": "not_applicable",
            "source_row_count": len(frame),
            "wide_row_count": len(frame),
        }
        data_layout = TRAINING_CURVE_LAYOUT
    value_columns = [
        column
        for column in frame.columns
        if column not in {"time", "step"} and pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = frame[value_columns].replace([np.inf, -np.inf], np.nan)
    summary = {column: {"mean": round(float(numeric[column].mean()), 4), "std": round(float(numeric[column].std()), 4), "min": round(float(numeric[column].min()), 4), "max": round(float(numeric[column].max()), 4), "missing_ratio": round(float(numeric[column].isna().mean()), 4)} for column in value_columns}
    visible: dict[str, Any] = {
        "columns": list(frame.columns),
        "row_count": len(frame),
        "series_count": k,
        "history_length_per_series": n,
        "frequency": "synthetic_step",
        "summary": summary,
        "business_constraints": {
            "latency": str(rng.choice(["low", "moderate", "relaxed"])),
            "interpretability": str(rng.choice(["required", "preferred"])),
            "compute_budget": str(rng.choice(["low", "medium", "high"])),
            "error_cost": str(rng.choice(["symmetric", "under_forecast_higher", "false_alarm_higher"])),
            "validation": "time_ordered",
        },
        "known_future_covariates": [],
        "data_layout": data_layout,
        "wide_column_map": wide_column_map,
    }
    primary_columns = [column for column in frame.columns if re.fullmatch(r"s\d+", str(column))]
    if primary_columns:
        primary_values = frame[primary_columns]
        visible["target_zero_ratio"] = round(float(primary_values.eq(0).sum().sum() / primary_values.notna().sum().sum()), 4) if primary_values.notna().sum().sum() else 0.0
    if pattern == "covariates":
        visible.update({
            "known_future_covariates": [column for column, item in wide_column_map["columns"].items() if item["source_column"] in {"promo_flag", "temperature"}],
            "unknown_future_covariates": [column for column, item in wide_column_map["columns"].items() if item["source_column"] == "actual_delivery_delay"],
        })
    if pattern == "hierarchical":
        visible["hierarchy"] = "region->store->series_id"
    if pattern == "interval":
        visible["forecast_requirement"] = "需要 50% 与 90% 预测区间并关注欠预测成本"
        truth["probabilistic_forecast_required"] = True

    component_hash = stable_hash(generator_truth)
    scenario_id = f"scenario_{index:05d}"
    manifest = {
        "id": scenario_id,
        "archetype": archetype["name"],
        "pattern": pattern,
        "task": archetype["task"],
        "task_goal": archetype["goal"],
        "series_count": archetype["count"],
        "history_length": archetype["length"],
        "recommended_mode": archetype["mode"],
        "image_value": archetype["image_value"],
        "analysis_difficulty": "medium",
        "difficulty": "medium",
        "visible_context": visible,
        "seed": seed,
        "generator_version": GENERATOR_VERSION,
        "data_layout": data_layout,
        "wide_column_map": wide_column_map,
        "signal_complexity": complexity,
        "component_hash": component_hash,
    }
    manifest["scenario_hash"] = stable_hash({"manifest": manifest, "truth": truth})
    return Scenario(manifest, frame, truth)


def _analysis_difficulty(scenario: Scenario, metrics: dict[str, Any] | None) -> tuple[str, list[str]]:
    truth = scenario.truth; generator = truth.get("generator", {}); pattern = truth.get("pattern")
    reasons: list[str] = []
    if generator.get("complexity") == "confounded": reasons.append("confounded_components")
    multi = generator.get("components", {}).get("multi_series", {})
    if metrics and metrics.get("within_group_mean") is not None and metrics.get("between_group_mean") is not None and abs(metrics["within_group_mean"] - metrics["between_group_mean"]) >= .15:
        reasons.append("cross_series_heterogeneity")
    events = generator.get("events", {})
    if pattern == "missing_blocks":
        mechanisms = {item.get("mechanism") for item in events.get("missingness", {}).values()}
        if any("value_dependent" in str(item) or "synchronous" in str(item) for item in mechanisms): reasons.append("ambiguous_missing_mechanism")
        if len(mechanisms) > 1: reasons.append("mixed_missing_mechanisms")
    if pattern in {"distribution_mismatch","sequence_anomaly","residual_hetero","drift"}: reasons.append("diagnostic_conflict")
    seasonal = generator.get("components", {}).get("signal", {}).get("seasonality", [])
    if pattern != "intermittent" and len(seasonal) >= 2 and any(item.get("period_drift") or item.get("regime_fraction") for item in seasonal): reasons.append("time_varying_multiseasonality")
    if pattern != "intermittent" and generator.get("components", {}).get("signal", {}).get("noise", {}).get("type") in {"student_t","heteroscedastic","level_dependent","mixture"}: reasons.append("nonstandard_noise")
    qualifying_conflict = any(item in reasons for item in ["confounded_components","cross_series_heterogeneity","ambiguous_missing_mechanism","diagnostic_conflict"])
    return ("hard" if qualifying_conflict and len(reasons) >= 2 else "medium"), reasons


def _refresh_hashes(scenario: Scenario) -> None:
    scenario.manifest["component_hash"] = stable_hash(scenario.truth["generator"])
    payload = dict(scenario.manifest); payload.pop("scenario_hash", None)
    scenario.manifest["scenario_hash"] = stable_hash({"manifest": payload, "truth": scenario.truth})


def create_scenario(index: int, archetype: dict[str, Any], seed: int, complexity: str = "compositional", multiseries_config: dict[str, Any] | None = None) -> Scenario:
    settings = validate_relation_config(multiseries_config)
    profile = sample_relation_profile(archetype["pattern"], seed, settings) if archetype["count"] == "multiple" else None
    failures: list[dict[str, Any]] = []
    for attempt in range(settings["max_attempts"] if settings["quality_gate"] else 1):
        scenario = _create_scenario_once(index, archetype, seed, complexity, profile, attempt)
        metrics = relation_metrics(scenario.frame, archetype["pattern"], scenario.truth)
        passed, reasons = relation_passes(archetype["pattern"], profile or {}, metrics)
        multi_truth = scenario.truth.get("generator", {}).get("components", {}).get("multi_series")
        if multi_truth is not None:
            multi_truth["realized_metrics"] = metrics
            multi_truth["quality_gate"] = {"passed":passed,"attempt":attempt + 1,"max_attempts":settings["max_attempts"],"failures":failures + ([{"attempt":attempt + 1,"reasons":reasons}] if reasons else [])}
        difficulty, difficulty_reasons = _analysis_difficulty(scenario, metrics)
        scenario.manifest["analysis_difficulty"] = difficulty; scenario.manifest["difficulty"] = difficulty
        scenario.manifest["difficulty_reasons"] = difficulty_reasons
        scenario.manifest["multiseries_quality_passed"] = bool(passed)
        _refresh_hashes(scenario)
        if passed or not settings["quality_gate"]:
            return scenario
        failures.append({"attempt":attempt + 1,"reasons":reasons,"metrics":metrics})
    raise RuntimeError(f"Unable to generate {archetype['name']} within multiseries quality bounds after {settings['max_attempts']} attempts: {failures[-1]['reasons']}")


def _save(fig: Any, path: Path) -> str:
    fig.tight_layout()
    fig.savefig(path, dpi=130, metadata={"Software": "api_sft"})
    plt.close(fig)
    return path.name


def _series_map(frame: pd.DataFrame, column: str = "value") -> dict[str, np.ndarray]:
    if "series_id" in frame:
        return {str(series_id): group.sort_values("time")[column].to_numpy() for series_id, group in frame.groupby("series_id")}
    wide_columns = _metric_wide_columns(frame, column)
    if wide_columns:
        ordered = frame.sort_values("time", kind="stable") if "time" in frame else frame
        return {name.split("__", 1)[0]: ordered[name].to_numpy() for name in wide_columns}
    if column in frame:
        return {"aggregate": frame.sort_values(frame.columns[0], kind="stable")[column].to_numpy()}
    raise KeyError(f"No observed columns found for metric {column}")


def _clean(values: np.ndarray) -> np.ndarray:
    return pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).interpolate().bfill().ffill().to_numpy()


def estimate_period(values: np.ndarray) -> int | None:
    clean = _clean(values)
    if len(clean) < 24 or float(np.std(clean)) < 1e-8:
        return None
    time = np.arange(len(clean), dtype=float)
    detrended = clean - np.polyval(np.polyfit(time, clean, 1), time)
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(detrended))
    valid = np.flatnonzero((frequencies > 0) & (frequencies >= 1 / max(4, len(clean) / 2)) & (frequencies <= .5))
    if not len(valid):
        return None
    best = int(valid[np.argmax(power[valid])])
    period = int(round(1 / frequencies[best]))
    return period if 2 <= period <= len(clean) // 2 else None


def estimate_anomaly_and_change(values: np.ndarray) -> tuple[list[int], int | None, dict[str, Any]]:
    clean = _clean(values)
    n = len(clean)
    window = max(5, min(31, n // 15))
    baseline = pd.Series(clean).rolling(window, center=True, min_periods=1).median().to_numpy()
    residual = clean - baseline
    median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - median))) + 1e-8
    robust_z = .6745 * (residual - median) / mad
    anomalies = np.flatnonzero(np.abs(robust_z) > 4.5).astype(int).tolist()
    candidates = range(max(4, int(n * .18)), min(n - 4, int(n * .82)))
    scores = [(index, abs(float(np.mean(clean[:index]) - np.mean(clean[index:]))) / (float(np.std(clean)) + 1e-8)) for index in candidates]
    change = max(scores, key=lambda item: item[1])[0] if scores and max(score for _, score in scores) > .35 else None
    return anomalies[:max(3, min(12, n // 50))], change, {"rolling_median_window": window, "robust_z_threshold": 4.5, "change_score_threshold": .35}


def render_images_from_observed(frame: pd.DataFrame, visible_context: dict[str, Any], out: Path) -> tuple[list[str], dict[str, Any]]:
    """Render figures using observed columns only; hidden scenario truth is not accepted."""
    out.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    provenance: dict[str, Any] = {}

    def record(name: str, method: str, parameters: dict[str, Any] | None = None) -> None:
        names.append(name)
        provenance[name] = {"method": method, "parameters": parameters or {}, "oracle_truth_used": False, "input_columns": list(frame.columns)}

    if "train_loss" in frame:
        fig, axis = plt.subplots(figsize=(9, 4.8))
        axis.plot(frame.step, frame.train_loss, label="train")
        axis.plot(frame.step, frame.validation_loss, label="validation")
        axis.set(title="Training and validation curves", xlabel="epoch", ylabel="loss")
        axis.legend(); axis.grid(alpha=.2)
        name = _save(fig, out / "training_curves.png"); record(name, "observed_training_curves")
        return names, provenance
    absolute_error_columns = _source_columns_for_metric(frame, "absolute_error")
    if absolute_error_columns:
        metric_groups = {metric: _source_columns_for_metric(frame, metric) for metric in ["absolute_error", "missing_rate", "feature_psi"]}
        aggregate = pd.DataFrame({"time": frame["time"]})
        for metric, columns in metric_groups.items():
            aggregate[metric] = frame[columns].mean(axis=1)
        aggregate = aggregate.groupby("time", sort=False).mean(numeric_only=True).reset_index()
        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        for axis, column in zip(axes, ["absolute_error", "missing_rate", "feature_psi"]):
            axis.plot(aggregate.time, aggregate[column]); axis.set_ylabel(column); axis.grid(alpha=.2)
        axes[-1].set_xlabel("time")
        name = _save(fig, out / "monitoring_drift.png"); record(name, "observed_group_mean_monitoring")
        return names, provenance
    residual_columns = _source_columns_for_metric(frame, "residual")
    if residual_columns:
        if "series_id" in frame:
            first_id = str(frame.series_id.iloc[0])
            first = frame[frame.series_id == first_id].sort_values("time")
            actual_values = first["actual"].to_numpy()
            prediction_values = first["prediction"].to_numpy()
            residual_values = first["residual"].to_numpy()
        else:
            residual_column = residual_columns[0]
            first_id = residual_column.split("__", 1)[0]
            ordered = frame.sort_values("time", kind="stable")
            actual_column = first_id if first_id in frame.columns else f"{first_id}__actual"
            prediction_column = f"{first_id}__prediction"
            actual_values = ordered[actual_column].to_numpy()
            prediction_values = ordered[prediction_column].to_numpy()
            residual_values = ordered[residual_column].to_numpy()
        residual = _clean(residual_values)
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        axes[0, 0].plot(actual_values, label="actual"); axes[0, 0].plot(prediction_values, label="prediction"); axes[0, 0].legend()
        axes[0, 1].plot(residual); axes[0, 1].axhline(0, color="black", lw=.7)
        lags = min(60, len(residual) // 3); axes[1, 0].bar(np.arange(lags + 1), acf(residual, nlags=lags))
        axes[1, 1].scatter(prediction_values, residual, s=8, alpha=.5)
        for axis, title in zip(axes.ravel(), ["Actual vs prediction", "Residual over time", "Residual ACF", "Residual vs fitted"]):
            axis.set_title(title); axis.grid(alpha=.2)
        name = _save(fig, out / "residual_diagnostics.png"); record(name, "observed_residual_diagnostics", {"series_id": str(first_id), "acf_lags": lags})
        return names, provenance

    series = _series_map(frame)
    fig, axis = plt.subplots(figsize=(10, 4.8))
    for series_id, values in series.items():
        axis.plot(values, lw=1, alpha=.8, label=series_id)
    axis.set(title="Time-series overview", xlabel="observation order", ylabel="value"); axis.grid(alpha=.2)
    if len(series) > 1: axis.legend(ncol=4, fontsize=7)
    name = _save(fig, out / "overview.png"); record(name, "observed_series_overview")

    all_values = np.concatenate([np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))] for values in series.values()])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8)); axes[0].hist(all_values, bins=35); axes[1].boxplot(all_values); axes[0].set_title("Distribution"); axes[1].set_title("Box plot")
    name = _save(fig, out / "distribution.png"); record(name, "observed_distribution", {"bins": 35})

    first_values = next(iter(series.values()))
    clean = _clean(first_values)
    lags = min(60, len(clean) // 3)
    frequencies = np.fft.rfftfreq(len(clean)); power = np.abs(np.fft.rfft(clean - clean.mean())) ** 2
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8)); axes[0].bar(np.arange(lags + 1), acf(clean, nlags=lags)); axes[0].set_title("ACF"); axes[1].plot(frequencies[1:], power[1:]); axes[1].set_title("FFT power spectrum")
    name = _save(fig, out / "acf_fft.png"); record(name, "observed_acf_fft", {"acf_lags": lags})

    estimated_period = estimate_period(first_values)
    if estimated_period and len(clean) >= 4 * estimated_period:
        result = STL(clean, period=estimated_period, robust=True).fit(); fig = result.plot(); fig.set_size_inches(10, 7)
        name = _save(fig, out / "stl.png"); record(name, "stl_with_observed_period_estimate", {"estimated_period": estimated_period, "robust": True})

    if len(series) > 1:
        columns = 3; row_count = math.ceil(len(series) / columns); fig, axes = plt.subplots(row_count, columns, figsize=(10, 2.5 * row_count)); axes = np.array(axes).reshape(-1)
        for axis, (series_id, values) in zip(axes, series.items()): axis.plot(values, lw=1); axis.set_title(series_id); axis.grid(alpha=.2)
        for axis in axes[len(series):]: axis.axis("off")
        name = _save(fig, out / "small_multiples.png"); record(name, "observed_small_multiples")
        fig, axis = plt.subplots(figsize=(10, 4.8)); normalized: dict[str,np.ndarray] = {}
        for series_id, values in series.items():
            observed = np.asarray(values,dtype=float); zscore = (observed - np.nanmean(observed)) / (np.nanstd(observed) + 1e-8); normalized[series_id] = zscore; axis.plot(zscore, lw=1, alpha=.7, label=series_id)
        axis.set_title("Normalized overlay"); axis.grid(alpha=.2)
        name = _save(fig, out / "normalized_overlay.png"); record(name, "observed_zscore_overlay", {"missing_handling":"preserve_nan_gaps","normalization":"observed_mean_std"})
        normalized_frame = pd.DataFrame(normalized); correlation = normalized_frame.corr(min_periods=max(8,len(normalized_frame)//10)); overlap = normalized_frame.notna().astype(int).T.dot(normalized_frame.notna().astype(int)); labels=list(correlation.columns)
        fig, axis = plt.subplots(figsize=(6, 5)); image = axis.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm"); fig.colorbar(image, ax=axis); axis.set_title("Pearson correlation (pairwise complete)"); axis.set_xticks(range(len(labels)),labels,rotation=45,ha="right"); axis.set_yticks(range(len(labels)),labels)
        name = _save(fig, out / "correlation_heatmap.png"); record(name, "observed_pairwise_complete_pearson", {"missing_handling":"pairwise_complete","minimum_overlap":max(8,len(normalized_frame)//10),"pairwise_overlap":overlap.to_dict()})
        first_id,second_id=labels[:2]; first,second=_clean(normalized[first_id]),_clean(normalized[second_id]); distance = cdist(first[:, None], second[:, None]); fig, axis = plt.subplots(figsize=(6, 5)); axis.imshow(distance, aspect="auto", cmap="viridis"); axis.set(title="DTW local cost matrix", xlabel=second_id, ylabel=first_id)
        name = _save(fig, out / "dtw_alignment.png"); record(name, "observed_pairwise_local_cost", {"distance":"euclidean","missing_handling":"linear_interpolation_then_edge_fill","series":[first_id,second_id]})

    if any(pd.Series(values).isna().any() for values in series.values()):
        pivot = pd.DataFrame(series).T
        fig, axis = plt.subplots(figsize=(10, 3.5)); axis.imshow(pivot.isna(), aspect="auto", cmap="Reds"); axis.set(title="Missingness heatmap", xlabel="time", ylabel="series")
        name = _save(fig, out / "missingness_heatmap.png"); record(name, "observed_missingness")

    anomalies, change, parameters = estimate_anomaly_and_change(first_values)
    fig, axis = plt.subplots(figsize=(10, 4.8)); axis.plot(first_values)
    if change is not None: axis.axvline(change, color="orange", ls="--", label="estimated change candidate")
    finite_anomalies = [index for index in anomalies if index < len(first_values) and np.isfinite(first_values[index])]
    if finite_anomalies: axis.scatter(finite_anomalies, np.asarray(first_values)[finite_anomalies], color="red", label="estimated anomaly candidate")
    axis.set_title("Estimated anomaly and change-point candidates"); axis.grid(alpha=.2)
    if change is not None or finite_anomalies: axis.legend()
    name = _save(fig, out / "anomaly_changepoint.png"); record(name, "robust_observed_candidate_detection", {**parameters, "estimated_change": change, "estimated_anomalies": finite_anomalies})
    return names, provenance


def render_images(scenario: Scenario, out: Path) -> list[str]:
    names, _ = render_images_from_observed(scenario.frame, scenario.manifest["visible_context"], out)
    return names


def persist_scenario(scenario: Scenario, root: Path) -> dict[str, Any]:
    out = root / scenario.manifest["id"]
    figures = out / "figures"
    out.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    for path in figures.glob("*.png"):
        path.unlink()
    scenario.frame.to_csv(out / "data.csv", index=False)
    write_json(out / "visible_context.json", scenario.manifest["visible_context"])
    write_json(out / "ground_truth.json", scenario.truth)
    image_names, provenance = render_images_from_observed(scenario.frame, scenario.manifest["visible_context"], figures)
    write_json(out / "figure_provenance.json", provenance)
    data_path = (out / "data.csv").resolve()
    truth_path = (out / "ground_truth.json").resolve()
    provenance_path = (out / "figure_provenance.json").resolve()
    image_paths = [(figures / name).resolve() for name in image_names]
    record = dict(scenario.manifest)
    record.update({
        "data_path": str(data_path),
        "truth_path": str(truth_path),
        "figure_provenance_path": str(provenance_path),
        "images": [str(path) for path in image_paths],
        "file_integrity": {
            "data_sha256": sha256_file(data_path),
            "truth_sha256": sha256_file(truth_path),
            "figure_provenance_sha256": sha256_file(provenance_path),
            "image_sha256": {path.name: sha256_file(path) for path in image_paths},
        },
    })
    write_json(out / "manifest.json", record)
    return record


def _scenario_files_exist(record: dict[str, Any]) -> bool:
    required = [record.get("data_path"), record.get("truth_path"), record.get("figure_provenance_path"), *record.get("images", [])]
    if not bool(required) or not all(path and Path(path).exists() for path in required):
        return False
    integrity = record.get("file_integrity") or {}
    expected = {
        record.get("data_path"): integrity.get("data_sha256"),
        record.get("truth_path"): integrity.get("truth_sha256"),
        record.get("figure_provenance_path"): integrity.get("figure_provenance_sha256"),
    }
    expected.update({path: (integrity.get("image_sha256") or {}).get(Path(path).name) for path in record.get("images", [])})
    return all(not digest or sha256_file(Path(path)) == digest for path, digest in expected.items() if path)


def _semantic_truth_hash(truth: dict[str, Any]) -> str:
    value = json.loads(json.dumps(truth, ensure_ascii=False))
    generator = value.get("generator") or {}
    generator.pop("version", None)
    generator.pop("migration", None)
    return stable_hash(value)


def _visible_context_for_migration(source: dict[str, Any], frame: pd.DataFrame, mapping: dict[str, Any]) -> dict[str, Any]:
    visible = json.loads(json.dumps(source.get("visible_context") or {}, ensure_ascii=False))
    numeric_columns = [
        column for column in frame.columns
        if column != "time" and pd.api.types.is_numeric_dtype(frame[column])
    ]
    numeric = frame[numeric_columns].replace([np.inf, -np.inf], np.nan)
    visible.update({
        "columns": list(frame.columns),
        "row_count": len(frame),
        "data_layout": WIDE_PANEL_LAYOUT,
        "wide_column_map": mapping,
        "summary": {
            column: {
                "mean": round(float(numeric[column].mean()), 4),
                "std": round(float(numeric[column].std()), 4),
                "min": round(float(numeric[column].min()), 4),
                "max": round(float(numeric[column].max()), 4),
                "missing_ratio": round(float(numeric[column].isna().mean()), 4),
            }
            for column in numeric_columns
        },
    })
    primary = [column for column in frame.columns if re.fullmatch(r"s\d+", str(column))]
    if primary:
        denominator = int(frame[primary].notna().sum().sum())
        visible["target_zero_ratio"] = round(float(frame[primary].eq(0).sum().sum() / denominator), 4) if denominator else 0.0
    for key in ["known_future_covariates", "unknown_future_covariates"]:
        source_names = set(source.get("visible_context", {}).get(key, []))
        if source_names:
            visible[key] = [name for name, item in mapping["columns"].items() if item["source_column"] in source_names]
    return visible


def validate_scenario_record(record: dict[str, Any]) -> dict[str, Any]:
    flags: list[str] = []
    if record.get("generator_version") != GENERATOR_VERSION:
        flags.append("wrong_generator_version")
    if record.get("data_layout") not in {WIDE_PANEL_LAYOUT, TRAINING_CURVE_LAYOUT}:
        flags.append("unknown_data_layout")
    if not _scenario_files_exist(record):
        flags.append("missing_or_corrupt_files")
        return {"passed": False, "flags": flags}
    frame = pd.read_csv(record["data_path"])
    truth = read_json(Path(record["truth_path"]))
    manifest_payload = {
        key: value for key, value in record.items()
        if key not in {"scenario_hash", "data_path", "truth_path", "figure_provenance_path", "images", "file_integrity"}
    }
    if record.get("component_hash") != stable_hash(truth.get("generator")):
        flags.append("component_hash_mismatch")
    if record.get("scenario_hash") != stable_hash({"manifest": manifest_payload, "truth": truth}):
        flags.append("scenario_hash_mismatch")
    visible = record.get("visible_context") or {}
    if list(frame.columns) != visible.get("columns") or len(frame) != int(visible.get("row_count", -1)):
        flags.append("visible_context_file_mismatch")
    if record.get("data_layout") == WIDE_PANEL_LAYOUT:
        if "series_id" in frame.columns or "time" not in frame.columns:
            flags.append("not_wide_panel")
        mapping = record.get("wide_column_map") or {}
        if mapping.get("version") != WIDE_PANEL_LAYOUT:
            flags.append("missing_wide_column_map")
        if int(mapping.get("wide_duplicate_time_rows", -1)) != int(frame.duplicated(["time"]).sum()):
            flags.append("duplicate_count_mismatch")
    provenance = read_json(Path(record["figure_provenance_path"]))
    if any(item.get("oracle_truth_used") is not False for item in provenance.values()):
        flags.append("oracle_figure_provenance")
    return {"passed": not flags, "flags": flags}


def migrate_scenarios(
    source_manifest: Path,
    target_dir: Path,
    limit: int | None = None,
    resume: bool = False,
) -> list[dict[str, Any]]:
    """Migrate legacy panels into the current independently validated scenario dataset."""

    sources = list(iter_jsonl(source_manifest))
    sources = sources[:limit] if limit else sources
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = _persisted_scenario_records(target_dir) if resume else {}
    rows: list[dict[str, Any]] = []
    try:
        for source in sources:
            previous = existing.get(source["id"])
            if previous and previous.get("migration", {}).get("source_scenario_hash") == source.get("scenario_hash") and validate_scenario_record(previous)["passed"]:
                rows.append(previous)
                continue
            long_frame = pd.read_csv(source["data_path"])
            if "series_id" in long_frame.columns:
                frame, mapping = to_wide_panel(long_frame)
                data_layout = WIDE_PANEL_LAYOUT
            else:
                frame = long_frame.copy()
                mapping = {
                    "version": TRAINING_CURVE_LAYOUT,
                    "time_column": "step" if "step" in frame.columns else str(frame.columns[0]),
                    "columns": {},
                    "duplicate_policy": "not_applicable",
                    "source_row_count": len(frame),
                    "wide_row_count": len(frame),
                }
                data_layout = TRAINING_CURVE_LAYOUT
            truth = read_json(Path(source["truth_path"]))
            before_truth_hash = _semantic_truth_hash(truth)
            truth.setdefault("generator", {})["version"] = GENERATOR_VERSION
            truth["generator"]["migration"] = {
                "source_generator_version": source.get("generator_version"),
                "source_scenario_hash": source.get("scenario_hash"),
                "conversion": WIDE_PANEL_LAYOUT if data_layout == WIDE_PANEL_LAYOUT else "identity",
            }
            if before_truth_hash != _semantic_truth_hash(truth):
                raise RuntimeError(f"Ground truth changed during migration: {source['id']}")
            manifest = {
                key: value for key, value in source.items()
                if key not in {"data_path", "truth_path", "figure_provenance_path", "images", "file_integrity", "scenario_hash", "component_hash"}
            }
            manifest.update({
                "generator_version": GENERATOR_VERSION,
                "data_layout": data_layout,
                "wide_column_map": mapping,
                "visible_context": _visible_context_for_migration(source, frame, mapping) if data_layout == WIDE_PANEL_LAYOUT else {
                    **source.get("visible_context", {}),
                    "columns": list(frame.columns),
                    "row_count": len(frame),
                    "data_layout": data_layout,
                    "wide_column_map": mapping,
                },
                "migration": {
                    "source_generator_version": source.get("generator_version"),
                    "source_scenario_hash": source.get("scenario_hash"),
                    "source_data_sha256": sha256_file(Path(source["data_path"])),
                    "semantic_truth_sha256": before_truth_hash,
                },
            })
            scenario = Scenario(manifest=manifest, frame=frame, truth=truth)
            metrics = relation_metrics(frame, str(source.get("pattern", "")), truth)
            profile = truth.get("generator", {}).get("components", {}).get("multi_series", {}).get("profile") or {}
            passed, reasons = relation_passes(str(source.get("pattern", "")), profile, metrics)
            if source.get("series_count") == "multiple" and not passed:
                raise RuntimeError(f"Current scenario relation quality failed for {source['id']}: {reasons}")
            scenario.manifest["multiseries_quality_passed"] = bool(passed)
            scenario.manifest["migration"]["quality_reasons"] = reasons
            _refresh_hashes(scenario)
            record = persist_scenario(scenario, target_dir)
            validation = validate_scenario_record(record)
            if not validation["passed"]:
                raise RuntimeError(f"Current scenario validation failed for {source['id']}: {validation['flags']}")
            rows.append(record)
            write_jsonl(target_dir / "scenarios.jsonl", rows)
    except Exception as exc:
        write_jsonl(target_dir / "scenarios.jsonl", rows)
        write_json(target_dir / "migration_report.json", {
            "status": "error",
            "generator_version": GENERATOR_VERSION,
            "source_manifest": str(source_manifest.resolve()),
            "requested_records": len(sources),
            "completed_records": len(rows),
            "error_type": type(exc).__name__,
            "error": str(exc),
        })
        raise
    write_jsonl(target_dir / "scenarios.jsonl", rows)
    panel_rows = [row for row in rows if row.get("data_layout") == WIDE_PANEL_LAYOUT]
    write_json(target_dir / "migration_report.json", {
        "status": "passed",
        "generator_version": GENERATOR_VERSION,
        "data_layout": WIDE_PANEL_LAYOUT,
        "source_manifest": str(source_manifest.resolve()),
        "source_manifest_sha256": sha256_file(source_manifest),
        "target_manifest_sha256": sha256_file(target_dir / "scenarios.jsonl"),
        "record_count": len(rows),
        "wide_panel_count": len(panel_rows),
        "training_curve_count": len(rows) - len(panel_rows),
        "source_duplicate_time_series_rows": sum(int(row.get("wide_column_map", {}).get("source_duplicate_time_series_rows", 0)) for row in panel_rows),
        "wide_duplicate_time_rows": sum(int(row.get("wide_column_map", {}).get("wide_duplicate_time_rows", 0)) for row in panel_rows),
        "checks": {
            "semantic_truth_preserved": True,
            "duplicate_counts_preserved": True,
            "relation_quality_passed": all(row.get("multiseries_quality_passed") is True for row in rows),
            "oracle_free_figure_provenance": True,
            "file_integrity_passed": all(validate_scenario_record(row)["passed"] for row in rows),
        },
    })
    return rows




def _scenario_index(scenario_id: str) -> int | None:
    prefix = "scenario_"
    suffix = scenario_id[len(prefix):] if scenario_id.startswith(prefix) else ""
    return int(suffix) if len(suffix) == 5 and suffix.isdigit() else None


def _persisted_scenario_records(output_dir: Path, count: int | None = None) -> dict[str, dict[str, Any]]:
    """Load complete V3 records already committed inside scenario directories."""

    records: dict[str, dict[str, Any]] = {}
    for manifest_path in sorted(output_dir.glob("scenario_*/manifest.json")):
        record = read_json(manifest_path)
        if not isinstance(record, dict):
            raise ValueError(f"Scenario manifest must contain an object: {manifest_path}")
        scenario_id = str(record.get("id", ""))
        index = _scenario_index(scenario_id)
        if index is None or manifest_path.parent.name != scenario_id:
            raise ValueError(f"Scenario manifest id does not match its directory: {manifest_path}")
        if count is not None and index > count:
            continue
        if record.get("generator_version") != GENERATOR_VERSION:
            continue
        if record.get("multiseries_quality_passed") is not True:
            continue
        if not _scenario_files_exist(record):
            continue
        records[scenario_id] = record
    return records


def _contiguous_scenario_records(records: dict[str, dict[str, Any]], count: int | None = None) -> list[dict[str, Any]]:
    """Return the longest complete scenario_00001..N prefix in canonical order."""

    upper = count if count is not None else max((_scenario_index(item) or 0 for item in records), default=0)
    rows: list[dict[str, Any]] = []
    for index in range(1, upper + 1):
        record = records.get(f"scenario_{index:05d}")
        if record is None:
            break
        rows.append(record)
    return rows


def rebuild_scenario_manifest(output_dir: Path, count: int | None = None) -> list[dict[str, Any]]:
    """Rebuild scenarios.jsonl from complete per-scenario manifests."""

    records = _persisted_scenario_records(output_dir, count)
    rows = _contiguous_scenario_records(records, count)
    write_jsonl(output_dir / "scenarios.jsonl", rows)
    return rows


def generate_scenarios(
    count: int,
    seed: int,
    output_dir: Path,
    resume: bool = False,
    task_ratios: dict[str, float] | None = None,
    generation_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    ratios = task_ratios or TASK_RATIOS
    if set(ratios) != set(TASK_RATIOS) or abs(sum(ratios.values()) - 1) > 1e-6:
        raise ValueError("Task ratios must contain all five tasks and sum to 1")
    settings = generation_config or {}
    version = str(settings.get("version", GENERATOR_VERSION))
    if version != GENERATOR_VERSION:
        raise ValueError(f"Only generator version {GENERATOR_VERSION} is supported")
    if not bool(settings.get("oracle_free_images", True)):
        raise ValueError(f"{GENERATOR_VERSION.upper()} requires oracle_free_images=true")
    mix = validate_complexity_mix(settings.get("complexity_mix", DEFAULT_COMPLEXITY_MIX))
    complexities = complexity_schedule(count, mix, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    by_task = {task: [archetype for archetype in ARCHETYPES if archetype["task"] == task] for task in ratios}
    manifest_path = output_dir / "scenarios.jsonl"
    existing: dict[str, dict[str, Any]] = {}
    if resume:
        if manifest_path.exists():
            existing.update(
                {
                    row["id"]: row
                    for row in iter_jsonl(manifest_path)
                    if (_scenario_index(str(row.get("id", ""))) or count + 1) <= count
                }
            )
        # A failed prior run may have committed complete per-scenario manifests
        # without reaching the final aggregate write. Directory manifests are
        # authoritative because their data, figures, and hashes were persisted
        # together.
        existing.update(_persisted_scenario_records(output_dir, count))
        write_jsonl(manifest_path, _contiguous_scenario_records(existing, count))
    raw = {task: count * ratio for task, ratio in ratios.items()}; quotas = {task: int(value) for task, value in raw.items()}
    for task, _ in sorted(raw.items(), key=lambda item: item[1] - int(item[1]), reverse=True)[:count - sum(quotas.values())]: quotas[task] += 1
    schedule: list[dict[str, Any]] = []
    while len(schedule) < count:
        for task in ratios:
            used = sum(archetype["task"] == task for archetype in schedule)
            if used < quotas[task]: schedule.append(by_task[task][used % len(by_task[task])])
    try:
        for index, (archetype, complexity) in enumerate(zip(schedule, complexities), 1):
            scenario_id = f"scenario_{index:05d}"
            candidate = create_scenario(index, archetype, seed + index, complexity, settings.get("multiseries"))
            previous = existing.get(scenario_id)
            semantic_match = False
            if previous and previous.get("truth_path") and Path(previous["truth_path"]).exists():
                semantic_match = _semantic_truth_hash(read_json(Path(previous["truth_path"]))) == _semantic_truth_hash(candidate.truth)
            reusable = bool(
                previous
                and previous.get("generator_version") == GENERATOR_VERSION
                and (previous.get("component_hash") == candidate.manifest["component_hash"] or semantic_match)
                and previous.get("multiseries_quality_passed") is True
                and _scenario_files_exist(previous)
            )
            record = previous if reusable else persist_scenario(candidate, output_dir)
            rows.append(record)
            existing[scenario_id] = record
    except Exception:
        # Preserve every complete scenario produced before a later quality-gate
        # failure. This makes a normal failed run immediately consumable and a
        # subsequent --resume restart from the true on-disk state.
        if resume:
            existing.update(_persisted_scenario_records(output_dir, count))
        write_jsonl(manifest_path, _contiguous_scenario_records(existing, count))
        raise
    write_jsonl(manifest_path, rows)
    return rows
