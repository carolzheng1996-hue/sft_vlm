#!/usr/bin/env python3
"""Generate a synthetic QA dataset for evaluating time-series modeling agents.

The dataset is designed to compare text-only reasoning with multimodal
reasoning over the same task. Each record contains a text-only prompt, a
multimodal prompt that points to a rendered chart, and a structured answer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DOMAINS = [
    "工业传感器监控",
    "智能电网负荷",
    "云服务容量指标",
    "电池健康管理",
    "金融交易监控",
    "电商流量与转化",
    "城市交通流量",
]


@dataclass
class Case:
    task_type: str
    capability: str
    domain: str
    question: str
    text_observation: str
    answer: dict[str, Any]
    metadata: dict[str, Any]
    plotter: Callable[[Path], None]
    difficulty_level: int = 2
    business_context: str = ""
    data_description: str = ""
    target: str = ""
    constraints: str = ""
    visual_assets: list[str] | None = None
    raw_data: dict[str, Any] | None = None


def round_series(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def indexed_series(values: np.ndarray, name: str = "value") -> list[dict[str, float | int]]:
    return [{"t": int(i), name: round(float(v), 4)} for i, v in enumerate(values)]


def raw_univariate(values: np.ndarray, name: str = "value") -> dict[str, Any]:
    return {
        "type": "univariate",
        "fields": ["t", name],
        "series": indexed_series(values, name),
    }


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def describe_stats(x: np.ndarray) -> str:
    first = x[: len(x) // 3]
    middle = x[len(x) // 3 : 2 * len(x) // 3]
    last = x[2 * len(x) // 3 :]
    return (
        f"长度={len(x)}，均值={np.mean(x):.2f}，标准差={np.std(x):.2f}，"
        f"最小值={np.min(x):.2f}，最大值={np.max(x):.2f}；"
        f"前三分之一均值={np.mean(first):.2f}，中段均值={np.mean(middle):.2f}，"
        f"后三分之一均值={np.mean(last):.2f}。"
    )


def save_single_series(
    path: Path,
    y: np.ndarray,
    title: str,
    ylabel: str = "value",
    markers: list[tuple[int, str]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.plot(np.arange(len(y)), y, color="#1f77b4", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    if markers:
        for idx, label in markers:
            ax.axvline(idx, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8)
            ax.text(idx + 1, np.max(y), label, color="#d62728", fontsize=8, va="top")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_prediction_plot(path: Path, y_true: np.ndarray, y_pred: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.plot(y_true, label="ground truth", color="#1f77b4", linewidth=1.9)
    ax.plot(y_pred, label="prediction", color="#ff7f0e", linewidth=1.7)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_multivariate_plot(path: Path, series: dict[str, np.ndarray], title: str) -> None:
    names = list(series)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.9 * len(names)), dpi=140, sharex=True)
    if len(names) == 1:
        axes = [axes]
    colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for ax, name, color in zip(axes, names, colors):
        ax.plot(series[name], color=color, linewidth=1.6)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.25)
    axes[0].set_title(title)
    axes[-1].set_xlabel("time step")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_interval_plot(
    path: Path,
    y_true: np.ndarray,
    median: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    title: str,
) -> None:
    t = np.arange(len(y_true))
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
    ax.fill_between(t, lower, upper, color="#aec7e8", alpha=0.55, label="80% interval")
    ax.plot(y_true, label="ground truth", color="#1f77b4", linewidth=1.9)
    ax.plot(median, label="median forecast", color="#ff7f0e", linewidth=1.6)
    ax.set_title(title)
    ax.set_xlabel("time step")
    ax.set_ylabel("value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def simple_acf(x: np.ndarray, max_lag: int) -> np.ndarray:
    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom == 0:
        return np.zeros(max_lag + 1)
    values = [1.0]
    for lag in range(1, max_lag + 1):
        values.append(float(np.dot(centered[:-lag], centered[lag:]) / denom))
    return np.array(values)


def save_residual_diagnostic_plot(
    path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    residual: np.ndarray,
    title: str,
    season: int = 7,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=140)
    axes[0, 0].plot(y_true, label="ground truth", color="#1f77b4", linewidth=1.4)
    axes[0, 0].plot(y_pred, label="prediction", color="#ff7f0e", linewidth=1.2)
    axes[0, 0].set_title("Forecast vs actual")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].plot(residual, color="#d62728", linewidth=1.2)
    axes[0, 1].axhline(0, color="#444444", linewidth=0.8)
    axes[0, 1].set_title("Residual over time")
    acf = simple_acf(residual, 35)
    axes[1, 0].bar(np.arange(len(acf)), acf, color="#2ca02c")
    for lag in (season, 2 * season, 3 * season):
        axes[1, 0].axvline(lag, color="#9467bd", linestyle="--", linewidth=0.8)
    axes[1, 0].set_title("Residual ACF")
    groups = [residual[np.arange(len(residual)) % season == i] for i in range(season)]
    axes[1, 1].boxplot(groups, tick_labels=[str(i + 1) for i in range(season)], showfliers=True)
    axes[1, 1].set_title("Residual by weekday")
    for ax in axes.ravel():
        ax.grid(True, alpha=0.22)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_small_multiples(path: Path, series: dict[str, np.ndarray], title: str) -> None:
    names = list(series)
    cols = 3
    rows = int(math.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 2.3 * rows), dpi=140, sharex=False)
    axes_arr = np.array(axes).reshape(-1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, (name, values) in enumerate(series.items()):
        ax = axes_arr[i]
        ax.plot(values, color=colors[i % len(colors)], linewidth=1.2)
        ax.set_title(name, fontsize=9)
        ax.grid(True, alpha=0.22)
    for ax in axes_arr[len(names) :]:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_model_recommendation_case(rng: np.random.Generator, idx: int) -> Case:
    n = int(rng.integers(168, 336))
    t = np.arange(n)
    weekly = 8 * np.sin(2 * np.pi * t / 7)
    daily = 2.5 * np.sin(2 * np.pi * t / 24 + rng.uniform(0, math.pi))
    drift_point = int(rng.integers(n // 3, 2 * n // 3))
    drift = np.where(t >= drift_point, rng.uniform(0.04, 0.12) * (t - drift_point), 0)
    pulses = np.zeros(n)
    pulse_positions = rng.choice(np.arange(12, n - 12), size=int(rng.integers(4, 9)), replace=False)
    pulses[pulse_positions] = rng.uniform(8, 18, size=len(pulse_positions))
    y = 50 + weekly + daily + drift + moving_average(pulses, 3) + rng.normal(0, 1.3, n)
    domain = random.choice(DOMAINS)
    recommended = "PatchTST 或 iTransformer，并加入节假日/业务事件等外生变量"
    if len(pulse_positions) >= 7:
        recommended = "N-BEATSx/TFT 或带事件特征的 PatchTST"
    question = (
        f"系统采集到一段{domain}核心指标历史数据。请分析该序列的动力学特征，"
        "并推荐用于未来 7 天预测的模型架构，说明理由。"
    )
    text = (
        describe_stats(y)
        + f" 周期性摘要：7 步滞后相关较强，24 步滞后相关中等；"
        + f"约在第 {drift_point} 个时间步之后均值缓慢抬升；"
        + f"检测到 {len(pulse_positions)} 个稀疏脉冲峰。"
    )

    def plotter(path: Path) -> None:
        save_single_series(path, y, f"Model recommendation case {idx}", "metric")

    return Case(
        "复杂时序特性的模型推荐",
        "意图识别与方案检索",
        domain,
        question,
        text,
        {
            "short_answer": recommended,
            "key_points": [
                "识别多周期叠加、概念漂移和局部脉冲",
                "避免只推荐通用 Transformer；需要局部 patch/变量交互/外生事件建模",
                "建议滚动回测并按漂移后窗口加权评估",
            ],
            "preferred_models": recommended,
        },
        {
            "pattern": "multi_period_drift_sparse_pulses",
            "drift_point": drift_point,
            "pulse_count": int(len(pulse_positions)),
        },
        plotter,
        raw_data={
            **raw_univariate(y, "metric"),
            "time_unit": "time_step",
            "known_events": {"drift_point": drift_point, "pulse_positions": [int(p) for p in sorted(pulse_positions)]},
        },
    )


def make_bad_case_case(rng: np.random.Generator, idx: int) -> Case:
    n = 168
    t = np.arange(n)
    base = 70 + 12 * np.sin(2 * np.pi * t / 24) + 4 * np.sin(2 * np.pi * t / 7)
    shock_start = int(rng.integers(60, 115))
    shock_width = int(rng.integers(6, 14))
    shock = np.zeros(n)
    shock[shock_start : shock_start + shock_width] = rng.uniform(16, 28)
    y_true = base + moving_average(shock, 3) + rng.normal(0, 1.2, n)
    failure_type = random.choice(["lag", "underestimate_peak", "phase_shift"])
    if failure_type == "lag":
        lag = int(rng.integers(1, 4))
        y_pred = np.r_[np.repeat(y_true[0], lag), y_true[:-lag]] + rng.normal(0, 0.8, n)
        diagnosis = "预测整体滞后，模型可能学习了复制上一时刻的平凡策略"
    elif failure_type == "underestimate_peak":
        y_pred = base + 0.35 * moving_average(shock, 3) + rng.normal(0, 1.0, n)
        diagnosis = "峰值被系统性低估，训练损失偏向均值且极端样本权重不足"
    else:
        shift = int(rng.integers(2, 5))
        y_pred = 70 + 12 * np.sin(2 * np.pi * (t - shift) / 24) + 4 * np.sin(2 * np.pi * t / 7)
        y_pred += 0.7 * moving_average(shock, 3) + rng.normal(0, 1.0, n)
        diagnosis = "周期相位错位，时间对齐或日历特征编码存在问题"
    err = np.abs(y_true - y_pred)
    worst = int(np.argmax(moving_average(err, 9)))
    domain = random.choice(DOMAINS)
    question = (
        f"这是某时序 foundation model 在{domain}预测任务上的真实值与预测值对比。"
        "请指出预测效果最差的区间，分析模型为何失效，并给出改进建议。"
    )
    text = (
        f"全局 MAE={np.mean(err):.2f}，最大绝对误差={np.max(err):.2f}；"
        f"第 {shock_start}-{shock_start + shock_width} 步附近出现业务冲击。"
        f"误差滑动均值峰值位于第 {max(0, worst - 4)}-{min(n - 1, worst + 4)} 步。"
    )

    def plotter(path: Path) -> None:
        save_prediction_plot(path, y_true, y_pred, f"Bad-case diagnosis {idx}")

    return Case(
        "预测结果 Bad Case 根因分析",
        "因果推断与根因分析",
        domain,
        question,
        text,
        {
            "short_answer": diagnosis,
            "worst_interval": [max(0, worst - 4), min(n - 1, worst + 4)],
            "key_points": [
                "定位真实值与预测值分离最明显的时间段",
                "判断滞后、峰值低估或相位错位，而不是只报告误差大",
                "建议加入事件特征、重采样极端峰值、使用加权/分位数损失或修正时间对齐",
            ],
        },
        {
            "failure_type": failure_type,
            "shock_start": shock_start,
            "shock_width": shock_width,
        },
        plotter,
        raw_data={
            "type": "forecast_comparison",
            "fields": ["t", "ground_truth", "prediction", "absolute_error"],
            "series": [
                {
                    "t": int(i),
                    "ground_truth": round(float(y_true[i]), 4),
                    "prediction": round(float(y_pred[i]), 4),
                    "absolute_error": round(float(err[i]), 4),
                }
                for i in range(n)
            ],
        },
    )


def make_multivariate_anomaly_case(rng: np.random.Generator, idx: int) -> Case:
    n = 180
    t = np.arange(n)
    trigger = int(rng.integers(65, 115))
    current = 20 + 3 * np.sin(2 * np.pi * t / 36) + rng.normal(0, 0.5, n)
    current[trigger : trigger + 8] += rng.uniform(9, 16)
    temp = 32 + 0.08 * t / n + rng.normal(0, 0.35, n)
    temp += 0.45 * np.r_[np.zeros(5), moving_average(np.maximum(current[:-5] - 23, 0), 9)]
    voltage = 3.72 - 0.006 * np.maximum(current - 23, 0) + rng.normal(0, 0.01, n)
    soh = 0.96 - 0.00008 * t + rng.normal(0, 0.001, n)
    fault = bool(np.max(temp[trigger + 5 : trigger + 25]) > 38)
    domain = "电池健康管理"
    question = (
        "监控系统发出告警。请结合电流、温度、电压和 SOH 的走势，判断这属于正常负载波动"
        "还是电池潜伏性故障；若是故障，说明可能触发起因。"
    )
    text = (
        f"电流在第 {trigger}-{trigger + 8} 步出现短时高负载；"
        f"温度峰值={np.max(temp):.2f}，峰值相对电流峰约滞后 5-10 步；"
        f"电压最低值={np.min(voltage):.3f}，SOH 整体斜率={np.polyfit(t, soh, 1)[0]:.6f}。"
    )
    label = "潜伏性热故障风险" if fault else "高负载下的可恢复波动"

    def plotter(path: Path) -> None:
        save_multivariate_plot(
            path,
            {
                "current": current,
                "temperature": temp,
                "voltage": voltage,
                "SOH": soh,
            },
            f"Multivariate anomaly localization {idx}",
        )

    return Case(
        "多变量关联与异常定位",
        "多模态感知与模式识别",
        domain,
        question,
        text,
        {
            "short_answer": label,
            "key_points": [
                "电流先激增，温度随后爬升，体现延迟相关而非同步噪声",
                "电压轻微下探与高负载一致，若温度持续升高则需判为热风险",
                "建议检查散热、内阻、采样对齐，并引入滞后交叉相关特征",
            ],
            "root_cause": "高电流脉冲后散热不足或内阻升高导致温度滞后上升",
        },
        {
            "trigger_step": trigger,
            "fault": fault,
            "lag_steps": "5-10",
        },
        plotter,
        raw_data={
            "type": "multivariate",
            "fields": ["t", "current", "temperature", "voltage", "SOH"],
            "series": [
                {
                    "t": int(i),
                    "current": round(float(current[i]), 4),
                    "temperature": round(float(temp[i]), 4),
                    "voltage": round(float(voltage[i]), 4),
                    "SOH": round(float(soh[i]), 6),
                }
                for i in range(n)
            ],
        },
    )


def make_changepoint_case(rng: np.random.Generator, idx: int) -> Case:
    n = int(rng.integers(160, 260))
    t = np.arange(n)
    cp = int(rng.integers(55, n - 55))
    level1 = rng.uniform(35, 70)
    level2 = level1 + rng.choice([-1, 1]) * rng.uniform(12, 28)
    y = np.where(t < cp, level1, level2)
    y += 4 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1.4, n)
    slope_after = rng.uniform(-0.05, 0.08)
    y[cp:] += slope_after * (t[cp:] - cp)
    domain = random.choice(DOMAINS)
    question = (
        f"{domain}监控指标疑似发生分布变化。请判断是否存在突变点，说明其形态，"
        "并给出模型训练或在线响应策略。"
    )
    text = (
        describe_stats(y)
        + f" 窗口均值检测显示第 {cp} 步前后均值差约 {abs(level2 - level1):.1f}；"
        + f"突变后趋势斜率约 {slope_after:.3f}。"
    )

    def plotter(path: Path) -> None:
        save_single_series(path, y, f"Change-point response {idx}", "metric", [(cp, "candidate")])

    return Case(
        "突变点检测与响应",
        "决策生成与自适应调整",
        domain,
        question,
        text,
        {
            "short_answer": f"存在明显突变点，约在第 {cp} 步",
            "key_points": [
                "区分水平跃迁与普通季节性波动",
                "建议采用变点检测、分段建模或漂移后窗口再训练",
                "在线预测应降低突变前数据权重，并监控残差是否持续偏移",
            ],
            "change_point": cp,
        },
        {"change_point": cp, "level_shift": float(level2 - level1)},
        plotter,
        raw_data={
            **raw_univariate(y, "metric"),
            "time_unit": "time_step",
            "known_events": {"candidate_change_point": cp},
        },
    )


def make_long_period_case(rng: np.random.Generator, idx: int) -> Case:
    n = 360
    t = np.arange(n)
    p1 = int(rng.choice([24, 30, 48]))
    p2 = int(rng.choice([90, 120, 168]))
    y = 100 + 7 * np.sin(2 * np.pi * t / p1) + 14 * np.sin(2 * np.pi * t / p2 + 0.4)
    y += rng.normal(0, 2.0, n)
    acf_p1 = float(np.corrcoef(y[:-p1], y[p1:])[0, 1])
    acf_p2 = float(np.corrcoef(y[:-p2], y[p2:])[0, 1])
    domain = random.choice(DOMAINS)
    question = (
        f"请根据{domain}序列识别主要长短周期，并判断更适合统计模型、频域增强模型"
        "还是长上下文深度模型。"
    )
    text = (
        describe_stats(y)
        + f" 候选 ACF：lag {p1}={acf_p1:.2f}，lag {p2}={acf_p2:.2f}；"
        + "短周期振幅较小，长周期振幅更大。"
    )

    def plotter(path: Path) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), dpi=140)
        axes[0].plot(y, color="#1f77b4", linewidth=1.4)
        axes[0].set_title(f"Long dependency period recognition {idx}")
        axes[0].grid(True, alpha=0.25)
        max_lag = min(180, n // 2)
        acf = [1.0]
        for lag in range(1, max_lag):
            acf.append(float(np.corrcoef(y[:-lag], y[lag:])[0, 1]))
        axes[1].bar(np.arange(max_lag), acf, color="#2ca02c", width=0.8)
        axes[1].axvline(p1, color="#d62728", linestyle="--", linewidth=1)
        axes[1].axvline(p2, color="#9467bd", linestyle="--", linewidth=1)
        axes[1].set_xlabel("lag")
        axes[1].set_ylabel("ACF")
        axes[1].grid(True, alpha=0.2)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    return Case(
        "长依赖周期识别",
        "模型推荐能力",
        domain,
        question,
        text,
        {
            "short_answer": f"主要周期为 {p1} 和 {p2}，应优先使用长上下文/频域增强模型",
            "key_points": [
                "同时识别短周期与长周期，不能只看最大局部峰",
                "若样本较短，SARIMA 可作基线，但深度模型需覆盖至少一个长周期上下文",
                "可选 TimesFM、PatchTST、FEDformer/TimesNet，并做多尺度输入",
            ],
            "periods": [p1, p2],
        },
        {"periods": [p1, p2], "acf": {str(p1): acf_p1, str(p2): acf_p2}},
        plotter,
        raw_data={
            **raw_univariate(y, "metric"),
            "time_unit": "time_step",
            "known_events": {"periods": [p1, p2]},
        },
    )


def make_interval_case(rng: np.random.Generator, idx: int) -> Case:
    n = 144
    t = np.arange(n)
    y_true = 45 + 8 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1.1, n)
    median = 45 + 8 * np.sin(2 * np.pi * (t - 1) / 24) + rng.normal(0, 0.6, n)
    mode = random.choice(["too_narrow", "too_wide_late", "miscalibrated_peak"])
    if mode == "too_narrow":
        width = np.full(n, rng.uniform(1.5, 2.5))
        diagnosis = "置信区间过窄，覆盖率不足"
    elif mode == "too_wide_late":
        width = np.linspace(2.0, rng.uniform(10, 16), n)
        diagnosis = "预测越往后区间过快变宽，不确定性传播过强"
    else:
        width = np.full(n, 4.0)
        spike = int(rng.integers(55, 105))
        y_true[spike : spike + 5] += rng.uniform(9, 15)
        diagnosis = "峰值区间校准失败，真实值刺破上界"
    lower = median - width
    upper = median + width
    coverage = float(np.mean((y_true >= lower) & (y_true <= upper)))
    domain = random.choice(DOMAINS)
    question = (
        f"以下是{domain}未来窗口的区间预测结果。请评估不确定性质量，判断区间是否合理，"
        "并提出校准或训练策略。"
    )
    text = (
        f"80% 区间经验覆盖率={coverage:.2f}；平均区间宽度={np.mean(upper - lower):.2f}；"
        f"后半段平均宽度/前半段平均宽度={np.mean((upper - lower)[n//2:]) / np.mean((upper - lower)[:n//2]):.2f}。"
    )

    def plotter(path: Path) -> None:
        save_interval_plot(path, y_true, median, lower, upper, f"Interval forecast evaluation {idx}")

    return Case(
        "区间预测评估",
        "结果分析与不确定性量化",
        domain,
        question,
        text,
        {
            "short_answer": diagnosis,
            "key_points": [
                "结合覆盖率、区间宽度和真实值是否频繁越界判断校准质量",
                "若过窄需校准分位数或增加异方差建模；若过宽需约束不确定性传播",
                "建议使用 conformal calibration、分位数损失重权或按预测步长分桶校准",
            ],
            "coverage": coverage,
        },
        {"interval_mode": mode, "coverage": coverage},
        plotter,
        raw_data={
            "type": "prediction_interval",
            "fields": ["t", "ground_truth", "median", "lower", "upper", "covered"],
            "series": [
                {
                    "t": int(i),
                    "ground_truth": round(float(y_true[i]), 4),
                    "median": round(float(median[i]), 4),
                    "lower": round(float(lower[i]), 4),
                    "upper": round(float(upper[i]), 4),
                    "covered": bool(lower[i] <= y_true[i] <= upper[i]),
                }
                for i in range(n)
            ],
        },
    )


def make_residual_diagnosis_case(rng: np.random.Generator, idx: int) -> Case:
    n = 210
    t = np.arange(n)
    weekday_bias = np.where(t % 7 == 0, 9.0, 0.0) + np.where(t % 7 >= 5, -3.0, 0.0)
    y_true = 180 + 18 * np.sin(2 * np.pi * t / 7) + 0.05 * t + rng.normal(0, 3.0, n)
    y_pred = 180 + 0.05 * t + 10 * np.sin(2 * np.pi * t / 7 + 0.2) + rng.normal(0, 2.0, n)
    residual = y_true - y_pred + weekday_bias
    y_true = y_pred + residual
    acf7 = float(simple_acf(residual, 21)[7])
    domain = random.choice(["电商流量与转化", "智能电网负荷", "城市交通流量"])
    question = (
        f"某团队已经在{domain}任务上训练了一个预测模型。请根据预测结果和残差诊断，"
        "判断模型是否已经充分拟合，并提出下一步改进方案。"
    )
    text = (
        f"残差均值={np.mean(residual):.2f}，标准差={np.std(residual):.2f}；"
        f"Ljung-Box 检验 p 值 < 0.01；残差 ACF 在 lag=7 约为 {acf7:.2f}；"
        "业务方反馈周一和周末误差更明显。"
    )

    def plotter(path: Path) -> None:
        save_residual_diagnostic_plot(path, y_true, y_pred, residual, f"Residual diagnosis {idx}")

    return Case(
        "残差诊断与遗漏因素发现",
        "结果分析能力",
        domain,
        question,
        text,
        {
            "short_answer": "模型残差不是白噪声，仍有周季节性和日历效应未被充分建模",
            "key_points": [
                "指出残差 ACF 在季节滞后显著，说明剩余结构未建模",
                "结合按星期残差分布识别周一/周末系统偏差",
                "建议加入日历、节假日、工作日类型、促销或天气等外生变量",
                "重新做 rolling backtest，并检查预测区间在高波动日期的覆盖率",
            ],
        },
        {"residual_acf_lag7": acf7, "systematic_weekday_bias": True},
        plotter,
        difficulty_level=3,
        business_context=f"{domain}模型已上线前验收，业务方关心系统性误差。",
        data_description="每日序列，含真实值、预测值、残差、残差 ACF 和按星期分布。",
        target="诊断模型误差来源，并给出可执行修正方案。",
        constraints="不能只看 MAE；需要判断残差是否接近白噪声。",
        visual_assets=["预测值 vs 真实值", "残差时间图", "残差 ACF", "按星期残差箱线图"],
        raw_data={
            "type": "residual_diagnostics",
            "fields": ["t", "ground_truth", "prediction", "residual", "weekday"],
            "series": [
                {
                    "t": int(i),
                    "ground_truth": round(float(y_true[i]), 4),
                    "prediction": round(float(y_pred[i]), 4),
                    "residual": round(float(residual[i]), 4),
                    "weekday": int(i % 7) + 1,
                }
                for i in range(n)
            ],
            "residual_acf_lag7": round(acf7, 4),
        },
    )


def make_intermittent_demand_case(rng: np.random.Generator, idx: int) -> Case:
    n = 240
    demand = np.zeros(n)
    sale_days = rng.choice(np.arange(n), size=int(rng.integers(22, 42)), replace=False)
    demand[sale_days] = rng.negative_binomial(3, 0.35, len(sale_days)) + 1
    demand[rng.choice(sale_days, size=3, replace=False)] += rng.integers(15, 35, 3)
    zero_ratio = float(np.mean(demand == 0))
    mean_interval = float(n / max(1, len(sale_days)))
    question = (
        "某维修备件需要预测未来 8 周需求量用于安全库存。该 SKU 销售极不连续。"
        "请推荐建模方法，并说明为什么常规 MAPE/深度点预测可能不合适。"
    )
    text = (
        f"历史长度={n} 天，零销量比例={zero_ratio:.2f}，非零销售日均需求={np.mean(demand[demand>0]):.2f}；"
        f"平均两次非零需求间隔约 {mean_interval:.1f} 天，偶尔出现批量需求。"
    )

    def plotter(path: Path) -> None:
        save_single_series(path, demand, f"Intermittent demand {idx}", "daily demand")

    return Case(
        "间歇性需求预测与指标选择",
        "模型推荐能力",
        "备件库存补货",
        question,
        text,
        {
            "short_answer": "优先考虑 Croston/TSB、零膨胀或负二项模型，并用 MASE、WAPE、pinball loss 等指标",
            "key_points": [
                "识别大量零值和偶发批量需求，不能套普通 MAPE",
                "区分需求发生概率和需求规模，可用 Croston、SBA、TSB 或零膨胀模型",
                "库存场景应关注服务水平、缺货成本和分位数预测",
                "用时间滚动验证，避免随机切分放大稀疏样本泄漏",
            ],
        },
        {"zero_ratio": zero_ratio, "mean_nonzero_interval": mean_interval},
        plotter,
        difficulty_level=2,
        business_context="维修备件库存补货，低频需求但缺货成本高。",
        data_description="单 SKU 日需求序列，多数日期为 0，少数日期出现批量需求。",
        target="预测未来 8 周需求并辅助安全库存决策。",
        constraints="需求稀疏，MAPE 在零值附近不可用，低估需求成本高。",
        visual_assets=["原始需求折线图"],
        raw_data={
            **raw_univariate(demand, "daily_demand"),
            "time_unit": "day",
            "zero_ratio": round(zero_ratio, 4),
            "nonzero_days": [int(i) for i in np.where(demand > 0)[0]],
        },
    )


def make_short_history_case(rng: np.random.Generator, idx: int) -> Case:
    n_new = 45
    n_old = 180
    t_old = np.arange(n_old)
    analogs: dict[str, np.ndarray] = {}
    for i in range(5):
        analogs[f"analog_{i+1}"] = 25 + i * 5 + 0.08 * t_old + 5 * np.sin(2 * np.pi * t_old / 7) + rng.normal(0, 2.0, n_old)
    t_new = np.arange(n_new)
    new_item = 30 + 0.12 * t_new + 4 * np.sin(2 * np.pi * t_new / 7) + rng.normal(0, 2.2, n_new)
    question = (
        "某新品只有 45 天日销量历史，但同类商品有 3 年历史。业务要求预测未来 28 天销量，"
        "请设计建模方案，而不是简单套单序列模型。"
    )
    text = (
        f"新品历史长度={n_new} 天；同类老品数量=5，最长历史约 3 年；"
        "新品和老品都存在周季节性，但新品水平和增长速度尚不稳定。"
    )

    def plotter(path: Path) -> None:
        save_small_multiples(path, {"new_item": new_item, **analogs}, f"Short history new product {idx}")

    return Case(
        "短历史新品预测方案设计",
        "方案设计能力",
        "新品销量预测",
        question,
        text,
        {
            "short_answer": "应使用相似品迁移、global model 或层级贝叶斯，而不是只训练新品单序列模型",
            "key_points": [
                "指出 45 天不足以稳定学习 28 天 horizon 和节假日效应",
                "利用同类商品历史做 global model、相似品检索或层级先验",
                "加入价格、渠道、曝光、促销等外生变量",
                "用冷启动分组回测评估新品泛化能力",
            ],
        },
        {"new_history_days": n_new, "analog_count": len(analogs)},
        plotter,
        difficulty_level=4,
        business_context="新品上市后需要快速形成补货预测，同类老品可提供迁移信息。",
        data_description="新品短历史 + 多个同类长历史 SKU 的日销量序列。",
        target="预测新品未来 28 天销量。",
        constraints="新品历史短，单序列模型容易过拟合；需要可落地的冷启动方案。",
        visual_assets=["新品与相似品 small multiples"],
        raw_data={
            "type": "multi_series_new_product",
            "time_unit": "day",
            "series_by_id": {
                "new_item": round_series(new_item),
                **{name: round_series(values) for name, values in analogs.items()},
            },
        },
    )


def make_promo_feature_case(rng: np.random.Generator, idx: int) -> Case:
    n = 365
    t = np.arange(n)
    promo_days = np.zeros(n)
    starts = rng.choice(np.arange(20, n - 20), size=8, replace=False)
    for s in starts:
        promo_days[s : s + int(rng.integers(2, 5))] = 1
    y = 120 + 12 * np.sin(2 * np.pi * t / 7) + 55 * promo_days + rng.normal(0, 8, n)
    leakage_feature = "未来 7 天是否有促销" if random.random() < 0.5 else "促销结束后 3 天汇总销量"
    question = (
        "某电商 SKU 促销日销量暴涨，团队准备用促销变量做未来 14 天预测。"
        "请说明特征工程方案，并判断候选特征中是否存在未来信息泄漏。"
    )
    text = (
        f"一年日销量数据，促销天数={int(np.sum(promo_days))}；促销日平均销量比非促销日高"
        f" {np.mean(y[promo_days==1]) - np.mean(y[promo_days==0]):.1f}。"
        f"候选特征包含：已排期促销标记、折扣力度、星期几、{leakage_feature}。"
    )

    def plotter(path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=140)
        ax.plot(y, color="#1f77b4", linewidth=1.4, label="sales")
        ax.scatter(np.where(promo_days == 1)[0], y[promo_days == 1], color="#d62728", s=18, label="promo")
        ax.set_title(f"Promotion impact and feature leakage {idx}")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    return Case(
        "促销外生变量与未来信息泄漏",
        "特征工程能力",
        "电商销量预测",
        question,
        text,
        {
            "short_answer": f"可使用预测时点已知的促销排期和折扣，但 `{leakage_feature}` 属于泄漏或不可用特征",
            "key_points": [
                "区分预测时点已知的日历/促销计划与未来真实结果",
                "加入促销强度、预热、滞后回落、库存/缺货标记等特征",
                "验证时按时间滚动，特征生成只使用 cutoff 之前可得信息",
                "促销期可单独评估加权误差或分位数风险",
            ],
        },
        {"leakage_feature": leakage_feature, "promo_days": int(np.sum(promo_days))},
        plotter,
        difficulty_level=3,
        business_context="电商促销驱动销量尖峰，业务已知未来部分促销排期。",
        data_description="日销量、促销标记、折扣力度、星期几和候选衍生特征。",
        target="预测未来 14 天 SKU 销量。",
        constraints="只能使用预测时点已知信息，不能引入未来真实销量或事后汇总。",
        visual_assets=["销量折线图", "促销日散点标注"],
        raw_data={
            "type": "sales_with_exogenous_flags",
            "fields": ["t", "sales", "promo_flag"],
            "series": [
                {"t": int(i), "sales": round(float(y[i]), 4), "promo_flag": int(promo_days[i])}
                for i in range(n)
            ],
            "candidate_leakage_feature": leakage_feature,
        },
    )


def make_validation_leakage_case(rng: np.random.Generator, idx: int) -> Case:
    n = 220
    test_start = int(rng.integers(160, 185))
    split_type = random.choice(["random_split", "global_scaler", "centered_rolling"])
    question = (
        "某团队汇报每日销量预测测试集 MAPE 很低。请判断其验证方案是否可靠，"
        "指出潜在数据泄漏，并给出正确的 backtesting 方案。"
    )
    if split_type == "random_split":
        text = "团队把全部日期随机打乱后做 80/20 train-test split，并在测试集上调参。"
        flaw = "随机打乱破坏时间顺序，未来样本会泄漏到训练分布和调参过程"
    elif split_type == "global_scaler":
        text = f"团队按时间切分，第 {test_start} 天后为测试集，但标准化均值和方差用全量数据拟合。"
        flaw = "归一化使用全量数据，测试期分布信息泄漏到训练过程"
    else:
        text = "团队构造 rolling mean 特征时使用 centered window，包含预测日之后的销量。"
        flaw = "centered rolling 特征使用未来观测，属于显式未来信息泄漏"

    def plotter(path: Path) -> None:
        fig, ax = plt.subplots(figsize=(9, 2.8), dpi=140)
        ax.broken_barh([(0, test_start)], (10, 8), facecolors="#1f77b4", label="train")
        ax.broken_barh([(test_start, n - test_start)], (10, 8), facecolors="#ff7f0e", label="test")
        if split_type == "random_split":
            random_points = rng.choice(np.arange(n), size=45, replace=False)
            ax.scatter(random_points, np.full_like(random_points, 25), s=12, color="#d62728", label="random test dates")
        ax.set_xlim(0, n)
        ax.set_ylim(5, 32)
        ax.set_yticks([])
        ax.set_xlabel("time step")
        ax.set_title(f"Validation leakage check {idx}")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    return Case(
        "验证方案与数据泄漏判断",
        "验证方案能力",
        "销量预测评估",
        question,
        text,
        {
            "short_answer": f"不可靠：{flaw}",
            "key_points": [
                "时序预测验证必须模拟真实预测时点，不能随机打乱或使用未来信息",
                "使用 expanding window 或 rolling forecast origin",
                "每个 fold 内独立拟合 scaler、编码器和特征工程",
                "调参与最终测试集需要分离，按 horizon 汇报 MAE/RMSE/MASE 等指标",
            ],
        },
        {"leakage_type": split_type, "test_start": test_start},
        plotter,
        difficulty_level=5,
        business_context="模型评估结果异常好，需要审计验证流程是否可信。",
        data_description="日销量建模流程说明和 train/test 切分方式。",
        target="发现泄漏风险并设计可靠回测。",
        constraints="主要考察逻辑推理；图像只辅助理解时间切分。",
        visual_assets=["训练/测试时间切分示意图"],
        raw_data={
            "type": "split_plan",
            "n_time_steps": n,
            "test_start": test_start,
            "split_type": split_type,
            "train_indices_time_holdout": list(range(0, test_start)),
            "test_indices_time_holdout": list(range(test_start, n)),
        },
    )


def make_model_comparison_case(rng: np.random.Generator, idx: int) -> Case:
    models = ["ARIMA", "Prophet", "LightGBM", "TFT"]
    mae = np.array([rng.uniform(95, 115), rng.uniform(90, 120), rng.uniform(80, 105), rng.uniform(78, 100)])
    rmse = mae + np.array([rng.uniform(20, 45), rng.uniform(25, 55), rng.uniform(30, 70), rng.uniform(35, 85)])
    latency = np.array([5, 18, 12, 95]) + rng.normal(0, 2, 4)
    explainability = ["高", "中高", "中", "低"]
    preferred = "LightGBM" if latency[3] > 80 and rmse[2] < rmse[3] + 12 else "TFT"
    question = (
        "库存补货场景中，多个模型的离线指标、稳定性、解释性和推理耗时不同。"
        "请不要只看单一指标，选择最适合上线的模型并说明权衡。"
    )
    rows = [
        f"{m}: MAE={mae[i]:.1f}, RMSE={rmse[i]:.1f}, latency={latency[i]:.1f}ms, explainability={explainability[i]}"
        for i, m in enumerate(models)
    ]
    text = "；".join(rows) + "。业务要求推理延迟 < 50ms，低估高峰需求成本较高。"

    def plotter(path: Path) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), dpi=140)
        axes[0].bar(models, mae, color="#1f77b4")
        axes[0].set_title("MAE")
        axes[1].bar(models, rmse, color="#ff7f0e")
        axes[1].set_title("RMSE")
        axes[2].bar(models, latency, color="#2ca02c")
        axes[2].axhline(50, color="#d62728", linestyle="--", linewidth=1)
        axes[2].set_title("Latency ms")
        for ax in axes:
            ax.tick_params(axis="x", rotation=25)
            ax.grid(True, axis="y", alpha=0.25)
        fig.suptitle(f"Model comparison {idx}")
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    return Case(
        "多模型上线选择与业务权衡",
        "模型比较能力",
        "库存补货预测",
        question,
        text,
        {
            "short_answer": f"首选 {preferred}，但需说明精度、延迟、解释性和高峰低估风险的权衡",
            "key_points": [
                "不能机械选择单个最低 RMSE/MAE",
                "结合延迟约束、解释性、维护成本和高峰期业务损失",
                "建议按高峰/低谷、SKU 分层、horizon 分桶做误差分析",
                "上线前保留 champion-challenger 和监控回滚机制",
            ],
        },
        {"preferred_model": preferred, "latency_limit_ms": 50},
        plotter,
        difficulty_level=3,
        business_context="库存补货模型上线评审，精度和服务延迟都重要。",
        data_description="四类模型的 MAE、RMSE、推理耗时、解释性和稳定性信息。",
        target="选择上线模型并给出权衡理由。",
        constraints="推理延迟 < 50ms；高峰需求低估成本高；模型需可维护。",
        visual_assets=["模型指标柱状图", "延迟阈值线"],
        raw_data={
            "type": "model_metric_table",
            "latency_limit_ms": 50,
            "models": [
                {
                    "model": models[i],
                    "MAE": round(float(mae[i]), 4),
                    "RMSE": round(float(rmse[i]), 4),
                    "latency_ms": round(float(latency[i]), 4),
                    "explainability": explainability[i],
                }
                for i in range(len(models))
            ],
        },
    )


def make_hierarchical_forecast_case(rng: np.random.Generator, idx: int) -> Case:
    n = 120
    t = np.arange(n)
    stores: dict[str, np.ndarray] = {}
    for city in ["A", "B"]:
        for s in range(3):
            base = rng.uniform(40, 90)
            stores[f"city_{city}_store_{s+1}"] = base + 8 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 4, n)
    city_a = sum(v for k, v in stores.items() if "city_A" in k)
    city_b = sum(v for k, v in stores.items() if "city_B" in k)
    total = city_a + city_b
    question = (
        "企业需要同时预测全国、省/市、门店销量，并要求上下级预测加总一致。"
        "请设计层级预测方案，包括模型、验证、指标和 reconciliation。"
    )
    text = (
        "层级结构：全国 -> 2 个城市 -> 每城 3 个门店；各门店有 120 天日销量。"
        "单独训练的门店、市级、全国模型预测结果存在加总不一致。"
    )

    def plotter(path: Path) -> None:
        preview = {"national": total, "city_A": city_a, "city_B": city_b}
        preview.update(stores)
        save_small_multiples(path, preview, f"Hierarchical forecasting {idx}")

    return Case(
        "层级预测与加总一致性",
        "方案设计能力",
        "连锁零售销量预测",
        question,
        text,
        {
            "short_answer": "应使用层级预测 reconciliation，如 bottom-up、top-down、middle-out 或 MinT",
            "key_points": [
                "识别上下级预测必须满足加总一致性",
                "比较 bottom-up、top-down、middle-out、MinT/OLS reconciliation 的适用性",
                "验证要按时间滚动，并在各层级分别评估 WAPE/MASE/服务水平",
                "可用 global model 共享门店信息，同时保留层级约束",
            ],
        },
        {"hierarchy": "national->2 cities->6 stores", "series_count": 9},
        plotter,
        difficulty_level=4,
        business_context="连锁零售需要全国、市级、门店级预测都可用于计划。",
        data_description="多层级日销量序列，底层门店加总到城市和全国。",
        target="设计加总一致的层级预测方案。",
        constraints="不同层级都要可解释、可评估，预测值上下级需要一致。",
        visual_assets=["层级序列 small multiples"],
        raw_data={
            "type": "hierarchical_series",
            "time_unit": "day",
            "hierarchy": {
                "national": ["city_A", "city_B"],
                "city_A": [k for k in stores if "city_A" in k],
                "city_B": [k for k in stores if "city_B" in k],
            },
            "series_by_id": {
                "national": round_series(total),
                "city_A": round_series(city_a),
                "city_B": round_series(city_b),
                **{name: round_series(values) for name, values in stores.items()},
            },
        },
    )


def make_online_monitoring_case(rng: np.random.Generator, idx: int) -> Case:
    n = 120
    t = np.arange(n)
    baseline_err = np.abs(rng.normal(8, 2, n))
    drift_start = int(rng.integers(82, 98))
    baseline_err[drift_start:] += np.linspace(5, 25, n - drift_start) + rng.normal(0, 2, n - drift_start)
    missing_rate = np.full(n, 0.01)
    missing_rate[drift_start:] += np.linspace(0.02, 0.14, n - drift_start)
    question = (
        "模型上线后最近两周误差显著升高。请设计线上监控与告警方案，判断可能原因，"
        "并说明何时触发重训练或回滚。"
    )
    text = (
        f"上线后监控 {n} 天；误差从第 {drift_start} 天后持续抬升；"
        f"最近两周 MAE={np.mean(baseline_err[-14:]):.1f}，上线初期两周 MAE={np.mean(baseline_err[:14]):.1f}；"
        f"缺失率最近升至 {np.max(missing_rate):.2f}。"
    )

    def plotter(path: Path) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), dpi=140, sharex=True)
        axes[0].plot(baseline_err, color="#d62728", linewidth=1.5)
        axes[0].axvline(drift_start, color="#444444", linestyle="--", linewidth=1)
        axes[0].set_ylabel("absolute error")
        axes[0].set_title(f"Online monitoring {idx}")
        axes[1].plot(missing_rate, color="#1f77b4", linewidth=1.5)
        axes[1].axvline(drift_start, color="#444444", linestyle="--", linewidth=1)
        axes[1].set_ylabel("missing rate")
        axes[1].set_xlabel("day after launch")
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

    return Case(
        "线上监控、漂移检测与重训练策略",
        "决策生成与自适应调整",
        "生产环境预测服务",
        question,
        text,
        {
            "short_answer": "应同时监控误差、数据质量、分布漂移和业务切片，并设置重训练/回滚阈值",
            "key_points": [
                "把误差升高与缺失率/输入分布变化联系起来排查",
                "监控 MAE/WAPE、coverage、残差偏移、缺失率、异常率、特征分布 PSI/KS",
                "按业务切片定位是否集中在某些门店/SKU/时段",
                "设置告警、自动回测、重训练、champion-challenger 和回滚策略",
            ],
        },
        {"drift_start": drift_start, "max_missing_rate": float(np.max(missing_rate))},
        plotter,
        difficulty_level=4,
        business_context="预测模型已上线，最近两周业务反馈误差明显变差。",
        data_description="上线后每日误差、缺失率和漂移迹象。",
        target="设计监控、告警、诊断、重训练和回滚方案。",
        constraints="需要可操作阈值和故障定位路径，不能只说重新训练。",
        visual_assets=["线上误差曲线", "缺失率曲线"],
        raw_data={
            "type": "online_monitoring",
            "fields": ["day", "absolute_error", "missing_rate"],
            "series": [
                {
                    "day": int(i),
                    "absolute_error": round(float(baseline_err[i]), 4),
                    "missing_rate": round(float(missing_rate[i]), 4),
                }
                for i in range(n)
            ],
        },
    )


CASE_BUILDERS = [
    make_model_recommendation_case,
    make_bad_case_case,
    make_multivariate_anomaly_case,
    make_changepoint_case,
    make_long_period_case,
    make_interval_case,
    make_residual_diagnosis_case,
    make_intermittent_demand_case,
    make_short_history_case,
    make_promo_feature_case,
    make_validation_leakage_case,
    make_model_comparison_case,
    make_hierarchical_forecast_case,
    make_online_monitoring_case,
]


def default_scoring(case: Case) -> dict[str, Any]:
    scoring: dict[str, Any] = {
        "base_total": 10,
        "base_criteria": {
            "数据特征识别": 2,
            "模型或方案合理性": 2,
            "验证与指标": 2,
            "结果分析深度": 2,
            "可执行性": 1,
            "风险意识": 1,
        },
    }
    scoring["optional_multimodal_bonus"] = {"图像信息利用": 2}
    return scoring


def build_record(case: Case, idx: int, image_path: Path, output_dir: Path) -> dict[str, Any]:
    image_rel = image_path.relative_to(output_dir).as_posix()
    business_context = case.business_context or f"{case.domain}时序建模任务"
    data_description = case.data_description or "见文本统计摘要和对应时序图"
    target = case.target or "完成时序建模分析并给出建议"
    constraints = case.constraints or "需要结合业务目标、数据特征、验证风险和落地成本"
    visual_assets = case.visual_assets or ["原始时序图"]
    scenario = (
        f"背景：{business_context}\n"
        f"数据：{data_description}\n"
        f"目标：{target}\n"
        f"约束：{constraints}"
    )
    text_prompt = scenario + "\n\n问题：\n" + case.question + "\n\n可用文本观测：\n" + case.text_observation
    multimodal_prompt = (
        scenario
        + "\n\n问题：\n"
        + case.question
        + "\n\n可用文本观测：\n"
        + case.text_observation
        + f"\n\n请同时参考图片：{image_rel}"
    )
    record = {
        "id": f"ts_agent_{idx:04d}",
        "task_type": case.task_type,
        "capability": case.capability,
        "difficulty_level": case.difficulty_level,
        "domain": case.domain,
        "benchmark_design": {
            "business_context": business_context,
            "data_description": data_description,
            "target": target,
            "constraints": constraints,
            "visual_assets": visual_assets,
        },
        "input": {
            "text_only": {
                "question": text_prompt,
                "modalities": ["text"],
            },
            "multimodal": {
                "question": multimodal_prompt,
                "modalities": ["text", "image"],
                "image": image_rel,
            },
        },
        "answer": case.answer,
        "rubric": [
            "是否正确识别主要时序模式或失败形态",
            "是否给出与模式匹配的模型/数据工程/校准建议",
            "是否结合业务目标、预测 horizon、可解释性、延迟和成本约束",
            "是否设计合理的 rolling/expanding backtesting 并避免数据泄漏",
            "是否避免把噪声误判为周期、把相关误判为因果",
            "是否明确说明证据来自统计文本、视觉曲线或二者结合",
        ],
        "scoring": default_scoring(case),
        "metadata": case.metadata,
    }
    if case.raw_data is not None:
        record["raw_data"] = case.raw_data
    return record


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# 时序建模 Agent Benchmark 摘要",
        "",
        f"- 样本数：{summary['num_samples']}",
        f"- 随机种子：{summary['seed']}",
        f"- 问答文件：`{summary['dataset']}`",
        f"- 图片目录：`{summary['image_dir']}/`",
        "",
        "## 任务分布",
        "",
        "| 任务类型 | 数量 |",
        "| --- | ---: |",
    ]
    for task_type, count in summary["task_type_counts"].items():
        lines.append(f"| {task_type} | {count} |")
    lines.extend(
        [
            "",
            "## 能力轴分布",
            "",
            "| 能力轴 | 数量 |",
            "| --- | ---: |",
        ]
    )
    for capability, count in summary["capability_counts"].items():
        lines.append(f"| {capability} | {count} |")
    lines.extend(
        [
            "",
            "## 难度层级分布",
            "",
            "| Level | 数量 |",
            "| ---: | ---: |",
        ]
    )
    for level, count in summary["difficulty_counts"].items():
        lines.append(f"| {level} | {count} |")
    lines.extend(
        [
            "",
            "## Benchmark 字段",
            "",
            "| 字段 | 说明 |",
            "| --- | --- |",
            "| `benchmark_design.business_context` | 业务背景 |",
            "| `benchmark_design.data_description` | 数据粒度、历史长度、变量或图形内容 |",
            "| `benchmark_design.target` | 预测、诊断或方案设计目标 |",
            "| `benchmark_design.constraints` | 可解释性、延迟、泄漏、成本等约束 |",
            "| `input.text_only.question` | 纯文本 prompt |",
            "| `input.multimodal.question` | 同题图文 prompt |",
            "| `input.multimodal.image` | 相对当前输出目录的图片路径 |",
            "| `answer` | 结构化参考答案 |",
            "| `rubric` | 通用评分要点 |",
            "| `scoring` | 基础 10 分评分维度和可选多模态加分 |",
            "| `metadata` | 合成数据真实标签 |",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_dataset(num_samples: int, output_dir: Path, seed: int) -> None:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records = []
    counts: dict[str, int] = {}
    capability_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    for idx in range(1, num_samples + 1):
        builder = CASE_BUILDERS[(idx - 1) % len(CASE_BUILDERS)]
        case = builder(rng, idx)
        image_path = image_dir / f"ts_agent_{idx:04d}.png"
        case.plotter(image_path)
        records.append(build_record(case, idx, image_path, output_dir))
        counts[case.task_type] = counts.get(case.task_type, 0) + 1
        capability_counts[case.capability] = capability_counts.get(case.capability, 0) + 1
        level_key = f"Level {case.difficulty_level}"
        difficulty_counts[level_key] = difficulty_counts.get(level_key, 0) + 1

    dataset_path = output_dir / "ts_agent_qa.jsonl"
    with dataset_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": num_samples,
        "seed": seed,
        "dataset": dataset_path.name,
        "image_dir": "images",
        "task_type_counts": counts,
        "capability_counts": capability_counts,
        "difficulty_counts": difficulty_counts,
        "schema": {
            "id": "unique sample id",
            "difficulty_level": "1-5 benchmark difficulty level",
            "benchmark_design": "business context, data description, target, constraints, visual assets",
            "input.text_only.question": "text-only prompt",
            "input.multimodal.question": "same prompt with image reference",
            "input.multimodal.image": "relative image path",
            "answer": "structured expected answer",
            "rubric": "scoring checklist",
            "scoring": "base 10-point scoring criteria plus optional multimodal bonus",
            "metadata": "synthetic ground-truth labels",
        },
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_summary_markdown(summary, output_dir / "summary.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=500, help="Number of QA pairs to generate.")
    parser.add_argument("--output-dir", type=Path, default=Path("dataset"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=20260707, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_dataset(args.num_samples, args.output_dir, args.seed)
    print(f"Generated {args.num_samples} QA pairs under {args.output_dir}")


if __name__ == "__main__":
    main()
