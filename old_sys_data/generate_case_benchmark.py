#!/usr/bin/env python3
"""Generate a case-level benchmark for time-series modeling agents.

Each case contains train/valid data, visible metadata, hidden generation labels,
figures, paired text-only/image-text questions, and answer keys.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CASE_TYPES = [
    "single_long_series",
    "multi_sku_panel",
    "intermittent_demand",
    "high_frequency_load",
    "sensor_anomaly_missing",
    "exogenous_leakage",
    "hierarchical_forecasting",
    "residual_diagnostics",
]

CASE_TYPE_CN = {
    "single_long_series": "单条长序列预测",
    "multi_sku_panel": "多 SKU panel 销售预测",
    "intermittent_demand": "间歇性需求预测",
    "high_frequency_load": "高频电力/流量预测",
    "sensor_anomaly_missing": "传感器异常与缺失",
    "exogenous_leakage": "外生变量与泄漏检测",
    "hierarchical_forecasting": "层级时间序列预测",
    "residual_diagnostics": "验证结果与残差诊断",
}

CHOICE_IDS = ["A", "B", "C", "D", "E"]


@dataclass
class CaseData:
    case_id: str
    case_type: str
    train_rows: list[dict[str, Any]]
    valid_rows: list[dict[str, Any]]
    metadata: dict[str, Any]
    config: dict[str, Any]
    data_dictionary: str
    predictions_rows: list[dict[str, Any]] | None = None
    metrics_rows: list[dict[str, Any]] | None = None


def date_range(start: str, periods: int, freq: str) -> list[datetime]:
    dt = datetime.fromisoformat(start)
    step = timedelta(days=1) if freq == "D" else timedelta(hours=1)
    return [dt + i * step for i in range(periods)]


def fmt_dt(dt: datetime, freq: str) -> str:
    return dt.strftime("%Y-%m-%d") if freq == "D" else dt.strftime("%Y-%m-%d %H:%M:%S")


def dow(dt: datetime) -> int:
    return dt.weekday()


def month(dt: datetime) -> int:
    return dt.month


def clamp_target(x: np.ndarray, integer: bool = True) -> np.ndarray:
    x = np.maximum(0, x)
    return np.rint(x).astype(int) if integer else x


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for row in rows:
            values = []
            for col in cols:
                value = row.get(col, "")
                if value is None:
                    values.append("")
                elif isinstance(value, float):
                    values.append(f"{value:.4f}")
                else:
                    values.append(str(value))
            f.write(",".join(values) + "\n")


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets = np.array([float(r["target"]) for r in rows if r.get("target") not in ("", None)], dtype=float)
    zero_ratio = float(np.mean(targets == 0)) if len(targets) else 0.0
    missing_ratio = 1.0 - len(targets) / max(1, len(rows))
    by_dow: dict[str, list[float]] = {str(i): [] for i in range(7)}
    for row in rows:
        if row.get("target") in ("", None):
            continue
        dt = datetime.fromisoformat(str(row.get("date") or row.get("timestamp")).split(" ")[0])
        by_dow[str(dow(dt))].append(float(row["target"]))
    return {
        "target_mean": round(float(np.mean(targets)), 4) if len(targets) else None,
        "target_std": round(float(np.std(targets)), 4) if len(targets) else None,
        "zero_ratio": round(zero_ratio, 4),
        "missing_ratio": round(missing_ratio, 4),
        "weekday_mean": {k: round(float(np.mean(v)), 4) if v else None for k, v in by_dow.items()},
    }


def group_series(rows: list[dict[str, Any]], time_col: str) -> dict[str, tuple[list[str], list[float]]]:
    grouped: dict[str, tuple[list[str], list[float]]] = {}
    for row in rows:
        sid = str(row.get("series_id") or row.get("store_id") or "series_001")
        if sid not in grouped:
            grouped[sid] = ([], [])
        if row.get("target") not in ("", None):
            grouped[sid][0].append(str(row[time_col]))
            grouped[sid][1].append(float(row["target"]))
    return grouped


def plot_overview(case: CaseData, out: Path) -> None:
    time_col = case.metadata["time_column"]
    rows = case.train_rows + case.valid_rows
    totals: dict[str, float] = {}
    split_time = case.valid_rows[0][time_col]
    for row in rows:
        if row.get("target") in ("", None):
            continue
        totals[str(row[time_col])] = totals.get(str(row[time_col]), 0.0) + float(row["target"])
    xs = list(range(len(totals)))
    ys = list(totals.values())
    split_idx = list(totals).index(str(split_time)) if str(split_time) in totals else len(case.train_rows)
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.3)
    ax.axvline(split_idx, color="#d62728", linestyle="--", linewidth=1, label="train/valid split")
    if case.config.get("has_level_shift") and case.config.get("level_shift_index") is not None:
        ax.axvline(case.config["level_shift_index"], color="#9467bd", linestyle=":", linewidth=1, label="level shift")
    ax.set_title(f"{case.case_id} overview")
    ax.set_xlabel("time index")
    ax.set_ylabel("total target")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "overview.png")
    plt.close(fig)


def plot_small_multiples(case: CaseData, out: Path) -> None:
    grouped = group_series(case.train_rows + case.valid_rows, case.metadata["time_column"])
    items = list(grouped.items())[:16]
    rows_n, cols_n = 4, 4
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(10, 7), dpi=140)
    for ax, (sid, (_, values)) in zip(axes.ravel(), items):
        ax.plot(values, linewidth=1.0)
        ax.set_title(sid, fontsize=8)
        ax.grid(True, alpha=0.2)
    for ax in axes.ravel()[len(items) :]:
        ax.axis("off")
    fig.suptitle(f"{case.case_id} small multiples")
    fig.tight_layout()
    fig.savefig(out / "small_multiples.png")
    plt.close(fig)


def plot_seasonal_profile(case: CaseData, out: Path) -> None:
    time_col = case.metadata["time_column"]
    dow_values: dict[int, list[float]] = {i: [] for i in range(7)}
    month_values: dict[int, list[float]] = {i: [] for i in range(1, 13)}
    hour_values: dict[int, list[float]] = {i: [] for i in range(24)}
    for row in case.train_rows:
        if row.get("target") in ("", None):
            continue
        dt = datetime.fromisoformat(str(row[time_col]))
        value = float(row["target"])
        dow_values[dow(dt)].append(value)
        month_values[month(dt)].append(value)
        hour_values[dt.hour].append(value)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
    axes[0].bar(range(7), [np.mean(dow_values[i]) if dow_values[i] else 0 for i in range(7)], color="#1f77b4")
    axes[0].set_title("weekday mean")
    axes[1].bar(range(1, 13), [np.mean(month_values[i]) if month_values[i] else 0 for i in range(1, 13)], color="#2ca02c")
    axes[1].set_title("monthly mean")
    axes[2].bar(range(24), [np.mean(hour_values[i]) if hour_values[i] else 0 for i in range(24)], color="#ff7f0e")
    axes[2].set_title("hourly mean")
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "seasonal_profile.png")
    plt.close(fig)


def plot_missingness(case: CaseData, out: Path) -> None:
    grouped = group_series(case.train_rows + case.valid_rows, case.metadata["time_column"])
    sids = list(grouped)[:40]
    times = sorted({str(r[case.metadata["time_column"]]) for r in case.train_rows + case.valid_rows})
    time_index = {t: i for i, t in enumerate(times)}
    matrix = np.zeros((len(sids), len(times)))
    for i, sid in enumerate(sids):
        sid_rows = [r for r in case.train_rows + case.valid_rows if str(r.get("series_id") or r.get("store_id") or "series_001") == sid]
        for row in sid_rows:
            if row.get("target") in ("", None):
                matrix[i, time_index[str(row[case.metadata["time_column"]])]] = 1
    fig, ax = plt.subplots(figsize=(10, 4.5), dpi=140)
    ax.imshow(matrix, aspect="auto", cmap="Reds", interpolation="nearest")
    ax.set_title(f"{case.case_id} missingness heatmap")
    ax.set_xlabel("time index")
    ax.set_ylabel("series")
    fig.tight_layout()
    fig.savefig(out / "missingness_heatmap.png")
    plt.close(fig)


def plot_anomaly(case: CaseData, out: Path) -> None:
    plot_overview(case, out)
    source = out / "overview.png"
    target = out / "anomaly_changepoint.png"
    target.write_bytes(source.read_bytes())


def plot_residual(case: CaseData, out: Path) -> None:
    if not case.predictions_rows:
        plot_overview(case, out)
        (out / "residual_diagnostics.png").write_bytes((out / "overview.png").read_bytes())
        return
    rows = case.predictions_rows
    actual = np.array([float(r["actual"]) for r in rows])
    pred_a = np.array([float(r["pred_model_a"]) for r in rows])
    residual = actual - pred_a
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=140)
    axes[0, 0].plot(actual, label="actual", linewidth=1.2)
    axes[0, 0].plot(pred_a, label="model_a", linewidth=1.1)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("actual vs prediction")
    axes[0, 1].plot(residual, color="#d62728", linewidth=1.1)
    axes[0, 1].axhline(0, color="#444", linewidth=0.8)
    axes[0, 1].set_title("residual over time")
    by_dow = [residual[np.arange(len(residual)) % 7 == i] for i in range(7)]
    axes[1, 0].boxplot(by_dow, tick_labels=[str(i + 1) for i in range(7)])
    axes[1, 0].set_title("residual by weekday")
    acf = [1.0]
    centered = residual - np.mean(residual)
    denom = float(np.dot(centered, centered))
    for lag in range(1, 29):
        acf.append(float(np.dot(centered[:-lag], centered[lag:]) / denom) if denom else 0)
    axes[1, 1].bar(range(len(acf)), acf, color="#2ca02c")
    axes[1, 1].axvline(7, color="#9467bd", linestyle="--")
    axes[1, 1].set_title("residual ACF")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "residual_diagnostics.png")
    plt.close(fig)


def generate_figures(case: CaseData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plot_overview(case, out)
    plot_small_multiples(case, out)
    plot_seasonal_profile(case, out)
    plot_missingness(case, out)
    plot_anomaly(case, out)
    plot_residual(case, out)


def make_rows_for_panel(
    dates: list[datetime],
    series_ids: list[str],
    rng: np.random.Generator,
    config: dict[str, Any],
    freq: str,
    domain: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = len(dates)
    shift_idx = config.get("level_shift_index")
    missing_start = config.get("missing_block_start")
    missing_len = config.get("missing_block_len", 0)
    for s_i, sid in enumerate(series_ids):
        base = rng.uniform(20, 120)
        trend_strength = rng.uniform(0.005, 0.06) if config.get("has_trend") else 0
        intermittent = sid in set(config.get("intermittent_series", []))
        short_start = config.get("short_history_starts", {}).get(sid, 0)
        for t, dt in enumerate(dates):
            if t < short_start:
                continue
            weekly = 8 * math.sin(2 * math.pi * dow(dt) / 7) if config.get("has_weekly_seasonality") else 0
            yearly = 10 * math.sin(2 * math.pi * dt.timetuple().tm_yday / 365) if config.get("has_yearly_seasonality") else 0
            daily = 12 * math.sin(2 * math.pi * dt.hour / 24) if config.get("has_daily_seasonality") else 0
            trend = trend_strength * t
            level_shift = config.get("level_shift_size", 0) if shift_idx is not None and t >= shift_idx else 0
            holiday = 1 if (dt.month == 12 and dt.day in (24, 25, 31)) or (dt.month == 1 and dt.day == 1) else 0
            promo = 1 if domain in {"retail_sales", "exogenous_leakage", "hierarchical_sales"} and (t % 61 in (0, 1, 2)) else 0
            price = round(float(base / 3 + rng.normal(0, 1.0)), 2)
            discount = 0.2 if promo else 0.0
            y = base + weekly + yearly + daily + trend + level_shift + holiday * 18 + promo * rng.uniform(15, 35)
            y += rng.normal(0, config.get("noise_scale", 3.0))
            if intermittent:
                occurs = rng.random() < config.get("demand_probability", 0.12)
                y = rng.lognormal(2.2, 0.8) if occurs else 0
            target: Any = clamp_target(np.array([y]))[0]
            if config.get("has_outliers") and t in config.get("outlier_indices", []):
                target = int(target + rng.integers(50, 120))
            if missing_start is not None and missing_start <= t < missing_start + missing_len and s_i % 4 == 0:
                target = ""
            stockout = 1 if promo and rng.random() < 0.12 else 0
            if stockout and target != "":
                target = int(float(target) * 0.55)
            row = {
                "date" if freq == "D" else "timestamp": fmt_dt(dt, freq),
                "series_id": sid,
                "target": target,
                "price": price,
                "discount": discount,
                "promo_flag": promo,
                "holiday_flag": holiday,
                "stockout_flag": stockout,
                "category": f"C{s_i % 4}",
                "region": ["East", "South", "North", "West"][s_i % 4],
            }
            if config.get("has_leakage_features"):
                row["target_t_plus_1"] = ""
                row["future_7d_mean_target"] = ""
                row["actual_delivery_delay"] = int(rng.integers(0, 5))
            rows.append(row)
    if config.get("has_leakage_features"):
        fill_leakage(rows, "date" if freq == "D" else "timestamp")
    return rows


def fill_leakage(rows: list[dict[str, Any]], time_col: str) -> None:
    by_sid: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_sid.setdefault(str(row["series_id"]), []).append(row)
    for sid_rows in by_sid.values():
        sid_rows.sort(key=lambda r: r[time_col])
        vals = [float(r["target"]) if r["target"] != "" else np.nan for r in sid_rows]
        for i, row in enumerate(sid_rows):
            nxt = vals[i + 1] if i + 1 < len(vals) else np.nan
            future = [v for v in vals[i + 1 : i + 8] if not np.isnan(v)]
            row["target_t_plus_1"] = "" if np.isnan(nxt) else round(float(nxt), 4)
            row["future_7d_mean_target"] = "" if not future else round(float(np.mean(future)), 4)


def split_rows(rows: list[dict[str, Any]], time_col: str, valid_steps: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    times = sorted({r[time_col] for r in rows})
    valid_times = set(times[-valid_steps:])
    train = [r for r in rows if r[time_col] not in valid_times]
    valid = [r for r in rows if r[time_col] in valid_times]
    return train, valid


def base_config(case_id: str, case_type: str, rng: np.random.Generator) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_type": case_type,
        "has_trend": True,
        "has_weekly_seasonality": True,
        "has_yearly_seasonality": bool(rng.random() < 0.65),
        "has_daily_seasonality": False,
        "has_multiple_seasonality": False,
        "has_level_shift": bool(rng.random() < 0.55),
        "level_shift_index": None,
        "level_shift_size": round(float(rng.uniform(12, 45)), 3),
        "has_outliers": True,
        "outlier_type": "event_spike",
        "has_missing_values": bool(rng.random() < 0.65),
        "missing_type": "block_missing",
        "missing_block_start": None,
        "missing_block_len": int(rng.integers(3, 10)),
        "has_leakage_features": False,
        "leakage_columns": [],
        "known_future_covariates": ["holiday_flag", "promo_flag", "price"],
        "unknown_future_covariates": ["stockout_flag"],
        "valid_validation_strategy": "rolling_origin_backtesting",
        "invalid_validation_strategies": ["random_split", "shuffle_split", "train_on_all_evaluate_on_train"],
        "suitable_metrics": ["MAE", "MASE", "pinball_loss"],
        "unsuitable_metrics": ["MAPE_for_intermittent_demand"],
        "recommended_model_strategies": ["global_gradient_boosting_with_lag_features", "use_known_future_covariates"],
        "red_flag_recommendations": ["random_train_test_split", "use_future_target_statistics", "delete_all_zero_sales"],
    }


def build_metadata(case_id: str, case_type: str, freq: str, n_steps: int, n_series: int, horizon: int) -> dict[str, Any]:
    domain_map = {
        "single_long_series": "single_series_business_metric",
        "multi_sku_panel": "retail_sales",
        "intermittent_demand": "spare_parts_inventory",
        "high_frequency_load": "electricity_or_traffic_load",
        "sensor_anomaly_missing": "industrial_sensor_monitoring",
        "exogenous_leakage": "retail_sales_leakage_audit",
        "hierarchical_forecasting": "hierarchical_sales",
        "residual_diagnostics": "forecast_result_diagnostics",
    }
    return {
        "case_id": case_id,
        "case_type": case_type,
        "case_type_name": CASE_TYPE_CN[case_type],
        "domain": domain_map[case_type],
        "frequency": freq,
        "time_column": "date" if freq == "D" else "timestamp",
        "target_column": "target",
        "id_column": "series_id",
        "history_length": n_steps,
        "num_series": n_series,
        "forecast_horizon": horizon,
        "business_goal": "forecasting_and_modeling_strategy_selection",
        "evaluation_focus": [
            "feature_identification",
            "validation_design",
            "leakage_detection",
            "metric_selection",
            "model_strategy",
        ],
        "known_future_covariates": ["holiday_flag", "promo_flag", "price"],
        "unknown_future_covariates": ["stockout_flag", "actual_delivery_delay"],
        "cost_asymmetry": "under_forecast_is_more_costly",
    }


def data_dictionary(metadata: dict[str, Any], config: dict[str, Any]) -> str:
    leak_cols = ", ".join(config.get("leakage_columns", [])) or "none"
    return f"""# Data Dictionary

- `{metadata['time_column']}`: time index.
- `series_id`: series identifier.
- `target`: value to forecast or diagnose.
- `price`: planned or listed price, normally known for scheduled retail forecasts.
- `discount`: planned discount rate.
- `promo_flag`: planned promotion indicator, known if promotion calendar is frozen.
- `holiday_flag`: calendar holiday indicator, known in the future.
- `stockout_flag`: actual stockout indicator, often unknown at forecast creation time.
- `category`, `region`: static series attributes.

Known future covariates: {', '.join(metadata['known_future_covariates'])}.
Unknown or risky future covariates: {', '.join(metadata['unknown_future_covariates'])}.
Hidden leakage columns in this case: {leak_cols}.
"""


def sample_case(case_id: str, case_type: str, seed: int) -> CaseData:
    rng = np.random.default_rng(seed)
    config = base_config(case_id, case_type, rng)
    freq = "D"
    n_steps = 420
    n_series = 1
    horizon = 28
    start = "2022-01-01"

    if case_type == "single_long_series":
        n_steps, n_series, horizon = 900, 1, 30
        config["has_yearly_seasonality"] = True
        config["has_level_shift"] = True
        config["level_shift_index"] = int(rng.integers(420, 650))
        config["outlier_indices"] = [int(rng.integers(650, 820))]
        series_ids = ["series_001"]
    elif case_type == "multi_sku_panel":
        n_steps, n_series, horizon = 420, 36, 28
        config["has_yearly_seasonality"] = True
        config["has_level_shift"] = True
        config["level_shift_index"] = int(rng.integers(220, 320))
        intermittent = [f"sku_{i:03d}" for i in range(1, n_series + 1) if i % 7 == 0]
        short = {f"sku_{i:03d}": int(rng.integers(180, 280)) for i in range(1, n_series + 1) if i % 9 == 0}
        config["intermittent_series"] = intermittent
        config["short_history_starts"] = short
        config["has_intermittent_series"] = True
        config["has_short_history_series"] = True
        config["recommended_model_strategies"] += ["sku_segmentation", "croston_or_tsb_for_intermittent_items"]
        series_ids = [f"sku_{i:03d}" for i in range(1, n_series + 1)]
    elif case_type == "intermittent_demand":
        n_steps, n_series, horizon = 360, 32, 56
        config["has_intermittent_series"] = True
        config["intermittent_series"] = [f"part_{i:03d}" for i in range(1, n_series + 1)]
        config["demand_probability"] = float(rng.uniform(0.05, 0.18))
        config["has_yearly_seasonality"] = False
        config["suitable_metrics"] = ["MAE", "MASE", "pinball_loss", "service_level_metric"]
        config["recommended_model_strategies"] = ["croston_or_tsb_for_intermittent_items", "zero_inflated_model", "probabilistic_inventory_forecast"]
        series_ids = [f"part_{i:03d}" for i in range(1, n_series + 1)]
    elif case_type == "high_frequency_load":
        freq, n_steps, n_series, horizon = "H", 24 * 90, 12, 48
        start = "2023-01-01T00:00:00"
        config["has_daily_seasonality"] = True
        config["has_weekly_seasonality"] = True
        config["has_yearly_seasonality"] = False
        config["has_multiple_seasonality"] = True
        config["known_future_covariates"] += ["temperature_forecast"]
        config["recommended_model_strategies"] += ["fourier_features", "multi_seasonal_model"]
        series_ids = [f"meter_{i:03d}" for i in range(1, n_series + 1)]
    elif case_type == "sensor_anomaly_missing":
        freq, n_steps, n_series, horizon = "H", 24 * 120, 20, 24
        start = "2023-01-01T00:00:00"
        config["has_daily_seasonality"] = True
        config["has_missing_values"] = True
        config["has_outliers"] = True
        config["has_wrong_unit_segment"] = True
        config["has_sensor_drift"] = True
        config["missing_block_start"] = int(rng.integers(400, 1300))
        config["missing_block_len"] = int(rng.integers(16, 48))
        series_ids = [f"sensor_{i:03d}" for i in range(1, n_series + 1)]
    elif case_type == "exogenous_leakage":
        n_steps, n_series, horizon = 420, 18, 28
        config["has_leakage_features"] = True
        config["leakage_columns"] = ["future_7d_mean_target", "target_t_plus_1", "actual_delivery_delay"]
        config["unknown_future_covariates"] += ["target_t_plus_1", "future_7d_mean_target"]
        series_ids = [f"sku_{i:03d}" for i in range(1, n_series + 1)]
    elif case_type == "hierarchical_forecasting":
        n_steps, n_series, horizon = 360, 24, 56
        config["requires_reconciliation"] = True
        config["hierarchy"] = "country->province->city->store->sku"
        config["recommended_model_strategies"] += ["forecast_reconciliation", "bottom_up_or_mint"]
        series_ids = [f"store_{i//3 + 1:02d}_sku_{i%3 + 1:02d}" for i in range(n_series)]
    elif case_type == "residual_diagnostics":
        n_steps, n_series, horizon = 420, 10, 28
        config["residual_has_weekly_autocorrelation"] = True
        config["coverage_90"] = round(float(rng.uniform(0.58, 0.72)), 3)
        config["best_metric_model_has_bad_coverage"] = True
        series_ids = [f"series_{i:03d}" for i in range(1, n_series + 1)]
    else:
        series_ids = ["series_001"]

    if config.get("has_level_shift") and config.get("level_shift_index") is None:
        config["level_shift_index"] = int(rng.integers(n_steps // 3, 2 * n_steps // 3))
    if config.get("has_missing_values") and config.get("missing_block_start") is None:
        config["missing_block_start"] = int(rng.integers(n_steps // 3, 2 * n_steps // 3))
    config["history_length"] = n_steps
    config["num_series"] = n_series
    config["forecast_horizon"] = horizon
    config["frequency"] = freq
    config["start_date"] = start

    dates = date_range(start, n_steps, freq)
    domain = build_metadata(case_id, case_type, freq, n_steps, n_series, horizon)["domain"]
    rows = make_rows_for_panel(dates, series_ids, rng, config, freq, domain)
    time_col = "date" if freq == "D" else "timestamp"
    train, valid = split_rows(rows, time_col, horizon)
    metadata = build_metadata(case_id, case_type, freq, n_steps, n_series, horizon)
    metadata["basic_statistics"] = summarize_rows(train)
    predictions, metrics = None, None
    if case_type == "residual_diagnostics":
        predictions, metrics = make_predictions_and_metrics(valid, rng, config, time_col)
    return CaseData(case_id, case_type, train, valid, metadata, config, data_dictionary(metadata, config), predictions, metrics)


def make_predictions_and_metrics(
    valid: list[dict[str, Any]], rng: np.random.Generator, config: dict[str, Any], time_col: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    sample = [r for r in valid if r.get("target") not in ("", None)][:280]
    for i, row in enumerate(sample):
        actual = float(row["target"])
        weekday_bias = 8 if i % 7 in (5, 6) else 0
        pred_a = actual - weekday_bias + rng.normal(0, 5)
        pred_b = actual + rng.normal(0, 9)
        width = 10
        rows.append(
            {
                time_col: row[time_col],
                "series_id": row["series_id"],
                "actual": round(actual, 4),
                "pred_model_a": round(float(pred_a), 4),
                "pred_model_b": round(float(pred_b), 4),
                "lower_90": round(float(pred_a - width), 4),
                "upper_90": round(float(pred_a + width), 4),
            }
        )
    residual = np.array([r["actual"] - r["pred_model_a"] for r in rows])
    mae_a = float(np.mean(np.abs(residual)))
    mae_b = float(np.mean([abs(r["actual"] - r["pred_model_b"]) for r in rows]))
    coverage = float(np.mean([(r["lower_90"] <= r["actual"] <= r["upper_90"]) for r in rows]))
    metrics = [
        {"model": "LightGBM", "MAE": round(mae_a, 4), "RMSE": round(mae_a * 1.35, 4), "coverage_90": round(coverage, 4), "training_time": 12.5},
        {"model": "TFT", "MAE": round(mae_b, 4), "RMSE": round(mae_b * 1.25, 4), "coverage_90": 0.86, "training_time": 95.0},
        {"model": "SeasonalNaive", "MAE": round(mae_a * 1.35, 4), "RMSE": round(mae_a * 1.7, 4), "coverage_90": 0.82, "training_time": 0.1},
    ]
    config["coverage_90"] = round(coverage, 4)
    return rows, metrics


def option_map(options: list[str]) -> dict[str, str]:
    return {CHOICE_IDS[i]: option for i, option in enumerate(options)}


def q_single(case: CaseData, qid: str, task_type: str, prompt: str, options: list[str], answer: str, explanation: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "case_id": case.case_id,
        "question_type": "single_choice",
        "task_type": task_type,
        "prompt": prompt,
        "options": option_map(options),
        "answer": [answer],
        "explanation": explanation,
    }


def q_multi(case: CaseData, qid: str, task_type: str, prompt: str, options: list[str], answers: list[str], explanation: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "case_id": case.case_id,
        "question_type": "multi_select",
        "task_type": task_type,
        "prompt": prompt + " 选择所有正确项。",
        "options": option_map(options),
        "answer": answers,
        "explanation": explanation,
        "scoring": "F1(selected_options, answer)",
    }


def q_rank(case: CaseData, qid: str, prompt: str, options: list[str], ranking: list[str], explanation: str) -> dict[str, Any]:
    return {
        "question_id": qid,
        "case_id": case.case_id,
        "question_type": "ranking",
        "task_type": "improvement_priority",
        "prompt": prompt,
        "options": option_map(options),
        "ideal_ranking": ranking,
        "explanation": explanation,
        "scoring": "Kendall tau or top-2 ordered accuracy",
    }


def generate_questions(case: CaseData) -> list[dict[str, Any]]:
    c = case.config
    questions = []
    feature_options = ["存在周季节性", "存在年度季节性", "存在永久 level shift", "数据严格平稳", "存在连续缺失块"]
    feature_ans = []
    if c.get("has_weekly_seasonality"):
        feature_ans.append("A")
    if c.get("has_yearly_seasonality"):
        feature_ans.append("B")
    if c.get("has_level_shift"):
        feature_ans.append("C")
    if c.get("has_missing_values"):
        feature_ans.append("E")
    questions.append(q_multi(case, f"{case.case_id}_q01", "feature_identification", "根据可见的 train/valid 数据、摘要或图像，下列哪些判断正确？", feature_options, feature_ans, "答案来自 generation_config 中的时序模式和缺失标签。"))

    quality_options = ["存在 block missing", "存在事件尖峰或点异常", "存在未来目标统计泄漏字段", "所有异常都应删除", "随机切分是合理验证方式"]
    quality_ans = []
    if c.get("has_missing_values"):
        quality_ans.append("A")
    if c.get("has_outliers"):
        quality_ans.append("B")
    if c.get("has_leakage_features"):
        quality_ans.append("C")
    questions.append(q_multi(case, f"{case.case_id}_q02", "data_quality", "下列哪些数据质量或建模风险存在？", quality_options, quality_ans, "缺失、异常和泄漏来自隐藏生成配置；删除所有异常和随机切分均为红旗。"))

    questions.append(q_single(case, f"{case.case_id}_q03", "model_strategy", "以下哪种建模策略最合理？", [
        "对每条序列单独训练 ARIMA，并随机切分验证",
        "使用 global LightGBM/CatBoost，加入滞后、滚动、日历和已知未来外生变量，并对特殊序列分组处理",
        "删除所有零值和促销尖峰后训练线性回归",
        "只使用最近 7 天均值，不做验证",
    ], "B", "多序列和复杂业务场景应优先考虑 global model、特征工程、分组处理和时间回测。"))

    leak_options = ["holiday_flag", "promo_flag 或 planned_promo", "future_7d_mean_target", "actual_delivery_delay", "day_of_week"]
    leak_ans = []
    if "future_7d_mean_target" in c.get("leakage_columns", []):
        leak_ans.append("C")
    if "actual_delivery_delay" in c.get("leakage_columns", []) or "actual_delivery_delay" in c.get("unknown_future_covariates", []):
        leak_ans.append("D")
    questions.append(q_multi(case, f"{case.case_id}_q04", "leakage_detection", f"目标是预测未来 {case.metadata['forecast_horizon']} 步 target，以下哪些字段可能导致数据泄漏？", leak_options, leak_ans, "未来目标统计和事后变量不可在预测时使用。"))

    questions.append(q_single(case, f"{case.case_id}_q05", "validation_design", "以下哪种验证方式最合理？", [
        "随机抽取 20% 样本作为验证集",
        "每条序列内部打乱后划分训练和验证",
        f"使用 rolling-origin backtesting，每次预测未来 {case.metadata['forecast_horizon']} 步",
        "在训练集上评估模型并汇报误差",
    ], "C", "时序预测验证必须模拟真实 cutoff 和 forecast horizon。"))

    questions.append(q_single(case, f"{case.case_id}_q06", "metric_selection", "以下哪个指标最不适合作为主要指标？", [
        "MAE",
        "MASE",
        "pinball loss",
        "MAPE，尤其在大量 target=0 的序列上",
    ], "D", "MAPE 在零值、低销量、截断销量和间歇需求场景中不稳定；普通场景下也需结合业务目标选择指标。"))

    questions.append(q_single(case, f"{case.case_id}_q07", "intermittent_demand", "如果发现部分序列零值比例很高，哪种处理最合理？", [
        "删除所有零销量样本",
        "使用 Croston/TSB、零膨胀或分组策略，并用 MASE/服务水平等指标评估",
        "统一使用 MAPE 作为主指标",
        "把零值全部填充为全局均值",
    ], "B", "间歇性需求需要专门模型或评估方式。"))

    questions.append(q_multi(case, f"{case.case_id}_q08", "residual_diagnosis", "根据 residual plot / ACF，如果残差在 lag=7 显著，下列哪些判断合理？", [
        "残差已经是白噪声",
        "模型可能遗漏了周季节性",
        "可以尝试加入 weekday 或日历特征",
        "应随机打乱数据重新训练",
        "需要按业务切片检查误差",
    ], ["B", "C", "E"], "残差自相关和 weekday 偏差说明模型仍有未解释结构。"))

    questions.append(q_single(case, f"{case.case_id}_q09", "prediction_interval", "如果 90% 预测区间覆盖率显著低于 90%，且高峰期真实值常超过上界，以下哪种解释最合理？", [
        "区间过宽",
        "模型低估了高峰期或特定场景下的不确定性",
        "只要 RMSE 低就没有问题",
        "该数据不能做任何预测",
    ], "B", "低覆盖率且上界被刺破说明区间校准不足，特别是高峰场景。"))

    questions.append(q_rank(case, f"{case.case_id}_q10", "当前模型在高峰、异常或特殊序列上误差较大。请按优先级排序以下改进措施。", [
        "检查 stockout/缺失/泄漏等数据质量问题",
        "加入促销、节假日、价格、weekday 等可用外生特征",
        "对间歇性或短历史序列分组建模/单独评估",
        "把训练集和验证集随机打乱重新划分",
    ], ["A", "B", "C", "D"], "先排查数据质量和泄漏，再改进特征与分组；随机打乱是最低优先级且不合理。"))
    return questions


def materialize_question_modes(case: CaseData, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    text_files = ["train.csv", "valid.csv", "metadata.json", "data_dictionary.md", "basic_statistics.json"]
    image_files = [
        "figures/overview.png",
        "figures/small_multiples.png",
        "figures/seasonal_profile.png",
        "figures/missingness_heatmap.png",
        "figures/anomaly_changepoint.png",
        "figures/residual_diagnostics.png",
    ]
    paired = []
    for q in questions:
        for mode in ["text_only", "image_text"]:
            item = dict(q)
            item["question_id"] = f"{q['question_id']}_{'text' if mode == 'text_only' else 'image'}"
            item["input_mode"] = mode
            item["visible_files"] = text_files if mode == "text_only" else text_files + image_files
            paired.append(item)
    return paired


def write_case(case: CaseData, root: Path) -> list[dict[str, Any]]:
    case_dir = root / "cases" / case.case_id
    figures_dir = case_dir / "figures"
    questions_dir = case_dir / "questions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    questions_dir.mkdir(parents=True, exist_ok=True)
    time_col = case.metadata["time_column"]
    write_csv(case_dir / "train.csv", case.train_rows)
    write_csv(case_dir / "valid.csv", case.valid_rows)
    if case.predictions_rows is not None:
        write_csv(case_dir / "valid_predictions.csv", case.predictions_rows)
    if case.metrics_rows is not None:
        write_csv(case_dir / "model_metrics.csv", case.metrics_rows)
    (case_dir / "metadata.json").write_text(json.dumps(case.metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "generation_config.json").write_text(json.dumps(case.config, ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "basic_statistics.json").write_text(json.dumps(summarize_rows(case.train_rows), ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "data_dictionary.md").write_text(case.data_dictionary, encoding="utf-8")
    generate_figures(case, figures_dir)
    base_questions = generate_questions(case)
    paired_questions = materialize_question_modes(case, base_questions)
    text_questions = [q for q in paired_questions if q["input_mode"] == "text_only"]
    image_questions = [q for q in paired_questions if q["input_mode"] == "image_text"]
    (questions_dir / "text_only_questions.json").write_text(json.dumps(text_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    (questions_dir / "image_text_questions.json").write_text(json.dumps(image_questions, ensure_ascii=False, indent=2), encoding="utf-8")
    answer_key = {
        q["question_id"]: {
            "question_type": q["question_type"],
            "answer": q.get("answer"),
            "ideal_ranking": q.get("ideal_ranking"),
            "explanation": q["explanation"],
        }
        for q in paired_questions
    }
    (case_dir / "answer_key.json").write_text(json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8")
    open_task = f"""# Open Task: {case.case_id}

你正在处理一个 {CASE_TYPE_CN[case.case_type]} 场景。

请基于 train.csv、valid.csv、metadata.json 和 data_dictionary.md：

1. 总结数据特征和主要风险；
2. 设计合理的验证方案；
3. 选择模型策略和指标；
4. 指出可能的数据泄漏或数据质量问题；
5. 若可见图像，请说明图像提供了哪些额外证据。
"""
    (questions_dir / "open_task.md").write_text(open_task, encoding="utf-8")
    return paired_questions


def write_scoring_rules(root: Path) -> None:
    scoring = {
        "single_choice": {"score": "1 if selected == answer[0] else 0"},
        "multi_select": {
            "recommended": "F1(selected_options, answer)",
            "alternative": "max(0, correct_hits/num_gold - false_positives/num_non_gold)",
        },
        "ranking": {
            "recommended": "Kendall tau or nDCG",
            "simple": "top-2 ordered accuracy",
        },
        "paired_evaluation": {
            "text_only": "visible_files excludes figures",
            "image_text": "same question with figures added",
            "multimodal_gain": "score(image_text) - score(text_only)",
        },
    }
    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "scoring_rules.json").write_text(json.dumps(scoring, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def generate_benchmark(output_dir: Path, cases_per_type: int, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    all_questions: list[dict[str, Any]] = []
    all_cases = []
    case_idx = 1
    for case_type in CASE_TYPES:
        for _ in range(cases_per_type):
            case_id = f"case_{case_idx:03d}"
            case_seed = seed + case_idx * 101
            case = sample_case(case_id, case_type, case_seed)
            questions = write_case(case, output_dir)
            all_questions.extend(questions)
            all_cases.append(
                {
                    "case_id": case.case_id,
                    "case_type": case.case_type,
                    "case_type_name": CASE_TYPE_CN[case.case_type],
                    "num_train_rows": len(case.train_rows),
                    "num_valid_rows": len(case.valid_rows),
                    "num_questions": len(questions),
                }
            )
            case_idx += 1
    questions_dir = output_dir / "questions"
    questions_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(questions_dir / "all_questions.jsonl", all_questions)
    write_jsonl(questions_dir / "text_only_questions.jsonl", [q for q in all_questions if q["input_mode"] == "text_only"])
    write_jsonl(questions_dir / "image_text_questions.jsonl", [q for q in all_questions if q["input_mode"] == "image_text"])
    write_scoring_rules(output_dir)
    case_counts = Counter(c["case_type"] for c in all_cases)
    question_counts = Counter(q["question_type"] for q in all_questions)
    mode_counts = Counter(q["input_mode"] for q in all_questions)
    summary = {
        "name": "TimeSeries-Agent-Benchmark case-level MVP",
        "num_cases": len(all_cases),
        "cases_per_type": cases_per_type,
        "num_questions": len(all_questions),
        "case_type_counts": dict(case_counts),
        "question_type_counts": dict(question_counts),
        "input_mode_counts": dict(mode_counts),
        "structure": {
            "cases": "cases/case_xxx/{train.csv, valid.csv, metadata.json, data_dictionary.md, generation_config.json, figures, questions, answer_key.json}",
            "questions": "questions/all_questions.jsonl plus text_only/image_text splits",
            "evaluation": "evaluation/scoring_rules.json",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# TimeSeries-Agent-Benchmark Case-Level MVP",
        "",
        f"- Cases: {len(all_cases)}",
        f"- Questions: {len(all_questions)}",
        f"- Cases per type: {cases_per_type}",
        "- Each case has train.csv, valid.csv, metadata, hidden generation config, figures, questions, and answer keys.",
        "",
        "## Case Types",
        "",
        "| Case type | Count |",
        "| --- | ---: |",
    ]
    for case_type, count in case_counts.items():
        lines.append(f"| {CASE_TYPE_CN[case_type]} (`{case_type}`) | {count} |")
    lines.extend(["", "## Question Types", "", "| Type | Count |", "| --- | ---: |"])
    for qtype, count in question_counts.items():
        lines.append(f"| {qtype} | {count} |")
    lines.extend(["", "## Input Modes", "", "| Mode | Count |", "| --- | ---: |"])
    for mode, count in mode_counts.items():
        lines.append(f"| {mode} | {count} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("timeseries_agent_benchmark_cases"))
    parser.add_argument("--cases-per-type", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_benchmark(args.output_dir, args.cases_per_type, args.seed)
    print(f"Wrote case-level benchmark to {args.output_dir}")


if __name__ == "__main__":
    main()
