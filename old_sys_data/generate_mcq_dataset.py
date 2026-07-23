#!/usr/bin/env python3
"""Convert the benchmark QA dataset into a multiple-choice evaluation set.

The generated options are task-aware. Distractors are intentionally plausible:
they usually contain a valid time-series idea, but miss a key pattern, violate a
business constraint, or introduce leakage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


CHOICE_IDS = ["A", "B", "C", "D"]


def stable_rng(text: str) -> random.Random:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def strip_instruction(prompt: str) -> str:
    replacements = [
        "请分析该序列的动力学特征，并推荐用于未来 7 天预测的模型架构，说明理由。",
        "请指出预测效果最差的区间，分析模型为何失效，并给出改进建议。",
        "请评估不确定性质量，判断区间是否合理，并提出校准或训练策略。",
        "请设计建模方案，而不是简单套单序列模型。",
    ]
    result = prompt
    for old in replacements:
        result = result.replace(old, "请从下列选项中选择最合适的一项。")
    return result


def as_choice(text: str, rationale: str, correct: bool = False) -> dict[str, Any]:
    return {"text": text, "rationale": rationale, "is_correct": correct}


def options_for_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    task = record["task_type"]
    meta = record.get("metadata", {})
    answer = record.get("answer", {})
    short = answer.get("short_answer", "")

    if task == "复杂时序特性的模型推荐":
        return [
            as_choice(
                "识别为多周期叠加、缓慢漂移和稀疏脉冲并存；优先用 PatchTST/iTransformer 或 TFT，并加入事件、日历等外生变量，采用滚动回测。",
                "覆盖了周期、漂移、局部脉冲、外生变量和验证方式。",
                True,
            ),
            as_choice(
                "用 SARIMA/Prophet 建模 7 步和 24 步季节项，再加一个线性趋势项，保持解释性并避免深度模型。",
                "季节和趋势处理合理，但对稀疏脉冲、局部形态和复杂非线性适配不足。",
            ),
            as_choice(
                "直接使用 TimesFM 零样本预测，不做外生变量或漂移后窗口加权，以减少训练成本。",
                "foundation model 可作为基线，但忽略事件变量和分布漂移适配。",
            ),
            as_choice(
                "用普通 LSTM/GRU 对原始序列端到端训练，只要窗口足够长即可自动学习所有模式。",
                "序列模型并非荒谬，但缺少对局部 patch、外生事件和漂移验证的处理。",
            ),
        ]

    if task == "预测结果 Bad Case 根因分析":
        failure = meta.get("failure_type")
        correct_text = {
            "lag": "最可能是整体预测滞后，模型在拐点附近像复制上一时刻状态；应修正时间对齐、加入领先外生变量并惩罚滞后误差。",
            "underestimate_peak": "最可能是峰值系统性低估，平均型损失压低极端冲击；应重权峰值样本、加入事件特征或使用分位数/加权损失。",
            "phase_shift": "最可能是周期相位错位，日历编码或采样对齐有问题；应检查时间戳、周期特征和预测 horizon 对齐。",
        }[failure]
        distractors = [
            ("主要是噪声方差过大，建议仅增大平滑窗口并降低模型复杂度。", "平滑可降噪，但会进一步抹平拐点或峰值，不能解释主要失败形态。"),
            ("主要是训练集太短，最优处理是扩大模型容量并延长训练轮数。", "容量可能有帮助，但没有定位滞后/峰值/相位的结构性问题。"),
            ("主要是异常点污染，建议删除误差最大的区间后重新训练。", "误差区间可能正是业务冲击，不应简单删除。"),
        ]
        return [as_choice(correct_text, short or "匹配元数据中的失败形态。", True)] + [
            as_choice(t, r) for t, r in distractors
        ]

    if task == "多变量关联与异常定位":
        if meta.get("fault"):
            correct = "电流先激增、温度滞后升高且电压下探，属于潜伏性热故障风险；应排查散热、内阻和采样对齐。"
        else:
            correct = "更像高负载下的可恢复波动：电流冲击后温度有滞后响应但未持续失控；仍需监控散热和交叉相关特征。"
        return [
            as_choice(correct, "同时利用多变量时间对齐、滞后相关和故障阈值。", True),
            as_choice("更应先判为温度传感器漂移：温度变化滞后于电流，可能是探头热惯性，应先用温度单变量残差阈值过滤。", "热惯性解释有一定合理性，但忽略了电流先行、电压响应和故障阈值联动。"),
            as_choice("更应判为 SOH 短时异常：SOH 是健康指标，应让 SOH 单变量模型主导告警，其余变量只作辅助展示。", "SOH 重要，但变化缓慢，不能解释短时电流-温度-电压联动。"),
            as_choice("更应拆成四个单变量告警器：这样可以降低交叉相关特征带来的误报，并便于维护。", "可维护性较好，但会丢失关键的延迟相关证据。"),
        ]

    if task == "突变点检测与响应":
        cp = meta.get("change_point")
        return [
            as_choice(
                f"应判为第 {cp} 步附近的持续性 level shift/regime change；训练时加入 regime 特征、分段建模或提高突变后窗口权重。",
                "区分了持续结构变化和孤立异常，并给出在线响应策略。",
                True,
            ),
            as_choice("应先把突变后 1-2 个窗口作为异常冷却期降权，继续用单一全历史模型，以减少分段模型维护成本。", "降权可能短期有用，但若是持久新状态，单一全历史模型会持续偏置。"),
            as_choice("应主要增加季节项和节假日项，因为均值变化可能来自季节强度改变，暂不引入 regime 特征。", "季节项可辅助解释，但不能充分处理持久水平跃迁。"),
            as_choice("应优先做 Box-Cox/标准化等方差稳定化处理，再用同一模型训练全部历史。", "方差稳定化可能辅助，但遗漏了主要的均值 regime shift。"),
        ]

    if task == "长依赖周期识别":
        periods = meta.get("periods", [])
        p_text = " 和 ".join(str(p) for p in periods)
        return [
            as_choice(
                f"主要周期应覆盖 {p_text}，尤其要保证上下文长度覆盖长周期；可用长上下文 Transformer、PatchTST 或频域增强模型，并以 SARIMA 作基线。",
                "同时考虑短周期、长周期、上下文长度和基线。",
                True,
            ),
            as_choice("只要 ACF 在短 lag 有波动，就选择低阶 ARMA，不需要显式处理长周期。", "ARMA 可能拟合局部相关，但会漏掉长依赖周期。"),
            as_choice("只建模最短周期即可，因为长周期通常是趋势项的一部分，可通过差分消除。", "把长周期当趋势差分会损失可预测结构。"),
            as_choice("直接做随机森林回归，只加入最近 7 个 lag，避免深度模型过拟合。", "短 lag 特征不足以覆盖长周期。",
            ),
        ]

    if task == "区间预测评估":
        mode = meta.get("interval_mode")
        correct = {
            "too_narrow": "区间偏窄且覆盖不足；应做分位数校准、conformal calibration 或异方差建模，而不是只优化点预测。",
            "too_wide_late": "后半段区间扩张过快；应按 horizon 分桶校准不确定性，约束过度扩散，同时检查递归误差传播。",
            "miscalibrated_peak": "正常期可能覆盖尚可，但峰值/促销期上界失效；应做条件化不确定性和场景分层校准。",
        }[mode]
        return [
            as_choice(correct, short or "匹配区间覆盖形态。", True),
            as_choice("先保留当前点预测模型，只在业务侧把安全系数提高一个固定比例来覆盖风险。", "固定安全系数能缓解风险，但没有校准不同 horizon 或场景下的不确定性。"),
            as_choice("用全局 conformal calibration 统一扩宽区间，不需要区分促销期、远期 horizon 或高波动切片。", "全局校准合理但不够细，可能掩盖场景条件化失配。"),
            as_choice("把越界样本标成噪声点降低权重，避免它们主导分位数模型。", "越界样本可能是关键风险场景，降权会让区间继续低估风险。"),
        ]

    if task == "残差诊断与遗漏因素发现":
        return [
            as_choice(
                "残差不是白噪声，ACF 在周季节 lag 显著且按星期有系统偏差；应加入日历/节假日/业务外生变量并重新 rolling backtest。",
                "正确利用残差时间图、ACF 和按星期分布。",
                True,
            ),
            as_choice("残差均值接近 0 且总体误差可控，可以先上线；后续只在监控中加入按星期 MAE 切片。", "监控切片有价值，但上线前已看到显著残差自相关和系统偏差。"),
            as_choice("主要应做残差后处理：对残差训练一个 AR 修正器，而不调整原模型特征。", "AR 修正器可能改善自相关，但未解释按星期系统偏差和业务日历缺失。"),
            as_choice("应把模型替换为更大的深度模型，并增加训练轮数，暂不加入人工日历特征以保持端到端。", "容量可能有帮助，但不能替代缺失的日历和外生变量。",
            ),
        ]

    if task == "间歇性需求预测与指标选择":
        return [
            as_choice(
                "应采用 Croston/SBA/TSB、零膨胀或负二项思路，分开建模需求发生与需求规模；指标用 MASE、WAPE、pinball loss 或服务水平。",
                "识别了零值稀疏、偶发批量需求和库存风险。",
                True,
            ),
            as_choice("使用 sMAPE 作为主指标，并用 Prophet/ETS 捕捉弱周季节性，避免复杂的零膨胀建模。", "sMAPE 比 MAPE 稳一些，但仍不能充分处理需求发生过程和库存服务水平。"),
            as_choice("用 ARIMA 对 log(销量+1) 建模，预测后取指数还原即可解决零值问题。", "变换可缓解偏态，但不能充分建模需求发生过程。"),
            as_choice("用全局 TFT 合并多个备件 SKU 训练，并继续用 RMSE 作为主指标。", "全局模型可利用跨 SKU 信息，但 RMSE 和点预测不一定匹配间歇库存风险。"),
        ]

    if task == "短历史新品预测方案设计":
        return [
            as_choice(
                "应做冷启动方案：利用同类商品做 global model、相似品迁移或层级贝叶斯，并加入价格/曝光/促销等外生变量，用新品回测评估。",
                "解决短历史、迁移信息和验证方式。",
                True,
            ),
            as_choice("用新品 45 天训练轻量 Prophet/ETS，并把同类商品只用于人工 sanity check，避免迁移偏差。", "轻量模型可作基线，但没有充分利用可迁移的同类长历史。"),
            as_choice("选一个最相似老品，按首周销量比例缩放其未来曲线，作为主预测方案。", "相似品缩放是强基线，但对新品增长、渠道和促销差异过于刚性。"),
            as_choice("先训练同类商品 global model，但验证只看老品普通切分，不专门做冷启动回测。", "global model 方向合理，但验证没有模拟新品冷启动场景。",
            ),
        ]

    if task == "促销外生变量与未来信息泄漏":
        leak = meta.get("leakage_feature")
        return [
            as_choice(
                f"应只使用预测时点已知且已冻结的促销排期、折扣、日历等特征；对 `{leak}` 必须检查是否在预测时可得，事后销量派生特征一律视为泄漏。",
                "核心是区分预测时可得信息和事后结果信息。",
                True,
            ),
            as_choice("为安全起见删除所有未来促销相关字段，仅保留历史促销滞后特征和星期几。", "能避免泄漏，但过度删除了预测时已知且业务关键的促销排期。"),
            as_choice("保留所有促销派生特征，但在 rolling CV 中观察是否过拟合；若指标稳定就认为没有泄漏。", "稳定指标不能证明特征在预测时可得，仍需逐字段做 cutoff 审计。"),
            as_choice("把促销日单独剔除训练非促销模型，再对促销期乘以历史平均 uplift。", "可作为基线，但会弱化促销强度、预热和库存等条件差异。"),
        ]

    if task == "验证方案与数据泄漏判断":
        return [
            as_choice(
                "当前评估不可靠；应模拟真实预测时点，用 rolling/expanding forecast origin，并在每个 fold 内独立拟合 scaler、编码器和滞后特征。",
                "覆盖时间顺序、泄漏控制和回测设计。",
                True,
            ),
            as_choice("改为最后 20% 时间段 holdout 即可；scaler 和目标编码器仍可用全量数据拟合以保持分布稳定。", "时间 holdout 是进步，但全量预处理仍泄漏测试期分布。"),
            as_choice("采用按月份分组的 K-fold CV，让同月样本不跨 fold，同时继续随机打乱月份顺序。", "分组有一定道理，但仍未模拟预测 cutoff 和 horizon。"),
            as_choice("使用 expanding window 回测，但所有滞后/滚动特征先在全量表上预计算以节省时间。", "回测框架正确，但全量预计算可能引入未来窗口信息。"),
        ]

    if task == "多模型上线选择与业务权衡":
        preferred = meta.get("preferred_model", "LightGBM")
        return [
            as_choice(
                f"优先选择 {preferred}，同时说明 MAE/RMSE、推理延迟、解释性、维护成本和高峰低估风险的权衡，并保留 challenger 监控。",
                "不只看单一离线指标，满足上线约束。",
                True,
            ),
            as_choice("选择 RMSE 最低的 TFT，并通过批量推理和缓存缓解延迟问题，暂不考虑解释性。", "批量优化可能有用，但若单次延迟约束严格且需解释，风险仍高。"),
            as_choice("选择最可解释的 Prophet/ARIMA，并在高峰期用规则修正，以降低维护成本。", "解释性和维护性好，但可能牺牲多变量和高峰预测能力。"),
            as_choice("四个模型做加权平均集成，以降低高峰低估风险，再通过异步服务规避延迟。", "集成可能稳定，但上线复杂度、延迟和解释性需要更严格验证。"),
        ]

    if task == "层级预测与加总一致性":
        return [
            as_choice(
                "应使用层级预测 reconciliation，如 bottom-up、top-down、middle-out 或 MinT，使全国、城市、门店预测加总一致，并分层评估。",
                "直接处理层级一致性和多层级评估。",
                True,
            ),
            as_choice("分别训练全国、城市、门店模型，最终采用各层级验证误差最低的预测，不强制加总一致。", "单层最优会造成计划口径冲突，不能满足一致性约束。"),
            as_choice("只做 top-down：全国预测后按历史平均占比分摊到门店，因为高层级噪声更低。", "top-down 是候选，但会压制门店局部模式，仍需分层验证。"),
            as_choice("只做 bottom-up：门店预测相加天然一致，因此不需要 MinT 或城市/全国层误差诊断。", "bottom-up 可行但未必最优，高层误差和噪声传播仍需评估。"),
        ]

    if task == "线上监控、漂移检测与重训练策略":
        return [
            as_choice(
                "应同时监控误差、覆盖率、数据质量、特征分布和业务切片；误差与缺失率同步升高时先定位数据问题，再触发重训、降级或回滚。",
                "覆盖监控指标、根因定位和处置策略。",
                True,
            ),
            as_choice("最近两周误差升高时先触发自动重训，并在新模型离线指标更好后替换线上模型。", "重训流程合理但少了数据质量、漂移根因和回滚检查。"),
            as_choice("主要监控整体 MAE、P95 延迟和服务可用性；业务切片等人工分析可以放在周报中处理。", "工程监控必要，但整体指标会掩盖局部漂移和输入质量问题。"),
            as_choice("保持当前模型不变，只提高告警阈值以减少波动误报，同时观察一个月再决策。", "降低误报有意义，但面对持续误差和缺失率上升会延误处置。"),
        ]

    return [
        as_choice(short or "选择与参考答案一致的分析。", "来自原始参考答案。", True),
        as_choice("采用更简单的传统模型并忽略外生变量。", "可能可作基线，但通常遗漏关键条件。"),
        as_choice("只根据单个指标做决策。", "忽略业务约束和诊断证据。"),
        as_choice("删除所有异常区间后重新训练。", "容易丢失关键业务场景。"),
    ]


def attach_choice_ids(record_id: str, options: list[dict[str, Any]]) -> tuple[list[dict[str, str]], str, dict[str, str]]:
    rng = stable_rng(record_id)
    shuffled = options[:]
    rng.shuffle(shuffled)
    choices: list[dict[str, str]] = []
    answer_key = ""
    distractor_analysis: dict[str, str] = {}
    for choice_id, option in zip(CHOICE_IDS, shuffled):
        choices.append({"id": choice_id, "text": option["text"]})
        if option["is_correct"]:
            answer_key = choice_id
        else:
            distractor_analysis[choice_id] = option["rationale"]
    if not answer_key:
        raise ValueError(f"No correct option for {record_id}")
    return choices, answer_key, distractor_analysis


def convert_record(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    options = options_for_record(record)
    choices, answer_key, distractor_analysis = attach_choice_ids(record["id"], options)
    choice_text = "\n".join(f"{c['id']}. {c['text']}" for c in choices)
    mcq_instruction = "\n\n请从以下 4 个选项中选择唯一最佳答案。注意：其他选项可能部分合理，但会遗漏关键证据、违反业务约束或引入评估风险。\n" + choice_text

    result = {
        "id": record["id"].replace("ts_agent_", "ts_agent_mcq_"),
        "source_id": record["id"],
        "task_type": record["task_type"],
        "capability": record["capability"],
        "difficulty_level": record.get("difficulty_level"),
        "domain": record.get("domain"),
        "benchmark_design": record.get("benchmark_design", {}),
        "input": {
            "text_only": {
                "question": strip_instruction(record["input"]["text_only"]["question"]) + mcq_instruction,
                "modalities": ["text"],
            },
            "multimodal": {
                "question": strip_instruction(record["input"]["multimodal"]["question"]) + mcq_instruction,
                "modalities": ["text", "image"],
                "image": record["input"]["multimodal"]["image"],
            },
        },
        "choices": choices,
        "answer_key": answer_key,
        "correct_choice": next(c for c in choices if c["id"] == answer_key),
        "explanation": next(o["rationale"] for o in options if o["is_correct"]),
        "distractor_analysis": distractor_analysis,
        "scoring": {
            "type": "single_choice_exact_match",
            "points": 1,
            "answer_field": "answer_key",
            "note": "选项均为近似合理方案，评分以唯一最佳答案 exact match 为准。",
        },
        "metadata": record.get("metadata", {}),
    }
    if "raw_data" in record:
        result["raw_data"] = record["raw_data"]
    return result


def write_summary(records: list[dict[str, Any]], output_dir: Path, seed: int) -> None:
    task_counts = Counter(r["task_type"] for r in records)
    capability_counts = Counter(r["capability"] for r in records)
    level_counts = Counter(f"Level {r['difficulty_level']}" for r in records)
    answer_counts = Counter(r["answer_key"] for r in records)
    summary = {
        "num_samples": len(records),
        "seed": seed,
        "dataset": "ts_agent_mcq.jsonl",
        "image_dir": "images",
        "task_type_counts": dict(task_counts),
        "capability_counts": dict(capability_counts),
        "difficulty_counts": dict(level_counts),
        "answer_key_counts": dict(answer_counts),
        "schema": {
            "raw_data": "original numeric series or structured numeric table, when available",
            "choices": "four single-choice options with IDs A-D",
            "answer_key": "correct option ID",
            "correct_choice": "correct option object",
            "distractor_analysis": "why each wrong option is plausible but not best",
            "scoring": "single-choice exact-match scoring",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# 选择题版时序建模 Agent Benchmark 摘要",
        "",
        f"- 样本数：{len(records)}",
        f"- 随机种子：{seed}",
        "- 问答文件：`ts_agent_mcq.jsonl`",
        "- 图片目录：`images/`",
        "- 评分方式：单选 exact match，每题 1 分",
        "",
        "## 任务分布",
        "",
        "| 任务类型 | 数量 |",
        "| --- | ---: |",
    ]
    for k, v in task_counts.items():
        lines.append(f"| {k} | {v} |")
    lines.extend(["", "## 答案位置分布", "", "| 选项 | 数量 |", "| --- | ---: |"])
    for key in CHOICE_IDS:
        lines.append(f"| {key} | {answer_counts.get(key, 0)} |")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_mcq(source_dir: Path, output_dir: Path, seed: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    source_records = read_jsonl(source_dir / "ts_agent_qa.jsonl")
    records = []
    for record in source_records:
        image_rel = record["input"]["multimodal"]["image"]
        shutil.copy2(source_dir / image_rel, output_dir / image_rel)
        records.append(convert_record(record, output_dir))
    with (output_dir / "ts_agent_mcq.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_summary(records, output_dir, seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("dataset_benchmark_v2"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_benchmark_mcq"))
    parser.add_argument("--seed", type=int, default=20260707)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_mcq(args.source_dir, args.output_dir, args.seed)
    print(f"Wrote MCQ dataset to {args.output_dir}")


if __name__ == "__main__":
    main()
