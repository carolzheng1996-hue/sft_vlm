#!/usr/bin/env python3
"""Create an HTML guide for the three benchmark datasets."""

from __future__ import annotations

import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "benchmark_datasets_guide.html"
V2 = ROOT / "dataset_benchmark_v2_with_series"
MCQ = ROOT / "dataset_benchmark_mcq_with_series"
CASES = ROOT / "timeseries_agent_benchmark_cases"


def esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def truncate(text: str, n: int = 780) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def counter_table(counter: Counter[str], label: str = "类型") -> str:
    rows = []
    max_v = max(counter.values()) if counter else 1
    for key, value in counter.items():
        rows.append(
            f"""
            <tr>
              <td>{esc(key)}</td>
              <td class="num">{value}</td>
              <td><div class="bar"><span style="width:{100 * value / max_v:.1f}%"></span></div></td>
            </tr>
            """
        )
    return f"<table><thead><tr><th>{label}</th><th class=\"num\">数量</th><th>相对规模</th></tr></thead><tbody>{''.join(rows)}</tbody></table>"


def compact_list(items: list[str], limit: int = 8) -> str:
    if len(items) <= limit:
        return "、".join(items)
    return "、".join(items[:limit]) + f" 等 {len(items)} 项"


def dataset_comparison_table(
    v2: list[dict[str, Any]],
    mcq: list[dict[str, Any]],
    case_summary: dict[str, Any],
) -> str:
    v2_tasks = sorted({r["task_type"] for r in v2})
    mcq_tasks = sorted({r["task_type"] for r in mcq})
    case_types = list(case_summary["case_type_counts"].keys())
    rows = [
        [
            "定位",
            "开放生成式问答，重点看推理质量和分析深度",
            "客观选择题，重点做自动化准确率评测",
            "完整项目式 case，重点测试 Agent 读取文件、理解数据、验证设计和多模态诊断",
        ],
        ["规模", f"{len(v2)} 条样本", f"{len(mcq)} 条样本", f"{case_summary['num_cases']} 个 case，{case_summary['num_questions']} 道 paired questions"],
        ["题型", "生成式问答", "单选题 exact match", "单选、多选、排序题"],
        ["输入形态", "JSONL；每条含 raw_data、prompt、image path、参考答案", "JSONL；每条含 raw_data、choices、answer_key、image path", "目录式；每个 case 含 train.csv、valid.csv、metadata、figures、questions、answer_key"],
        ["任务覆盖", compact_list(v2_tasks), compact_list(mcq_tasks), compact_list(case_types)],
        ["图像", "每条样本 1 张任务图", "每条样本 1 张任务图", "每个 case 6 张诊断图"],
        ["原始序列", "有：raw_data", "有：raw_data", "有：train.csv / valid.csv"],
        ["标准答案", "结构化 answer + rubric，偏主观", "answer_key，客观", "answer_key；多选/排序有 scoring rules"],
        ["推荐用途", "质检、案例分析、开放式能力评估", "模型排行榜、text-only vs image-text 准确率对比", "端到端 Agent benchmark、项目式评测、多文件输入评测"],
    ]
    body = "\n".join(
        f"<tr><th>{esc(name)}</th><td>{esc(a)}</td><td>{esc(b)}</td><td>{esc(c)}</td></tr>" for name, a, b, c in rows
    )
    return f"""
    <table class="compare-table">
      <thead>
        <tr>
          <th>维度</th>
          <th>dataset_benchmark_v2_with_series</th>
          <th>dataset_benchmark_mcq_with_series</th>
          <th>timeseries_agent_benchmark_cases</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
    """


def detail_intro_sections(
    v2: list[dict[str, Any]],
    mcq: list[dict[str, Any]],
    case_summary: dict[str, Any],
) -> str:
    v2_task = Counter(r["task_type"] for r in v2)
    raw_types = Counter(r["raw_data"]["type"] for r in v2)
    diff = Counter(f"Level {r['difficulty_level']}" for r in v2)
    mcq_answers = Counter(r["answer_key"] for r in mcq)
    return f"""
    <section>
      <h2>每个数据集的详细简介</h2>
      <div class="dataset-detail">
        <h3>1. dataset_benchmark_v2_with_series</h3>
        <p>这是增强版生成式问答数据集，共 <strong>{len(v2)}</strong> 条样本。它保留了任务文本、图像、多模态输入路径、结构化参考答案，并新增 <code>raw_data</code>，因此既能构造纯文本/数值输入，也能构造图文输入。</p>
        <p><strong>任务类型：</strong>{esc(compact_list(list(v2_task.keys()), 14))}。</p>
        <p><strong>数据分布：</strong>任务类型基本均衡，前 10 类各 36 条，后 4 类各 35 条；难度分布为 {esc(dict(diff))}。</p>
        <p><strong>raw_data 分布：</strong>{esc(dict(raw_types))}。</p>
        <p><strong>适合评测：</strong>模型是否能给出完整、可执行、符合业务约束的时序建模分析。因为答案是生成式，建议用于人工评分、rubric 评分或 LLM-as-judge 辅助评分。</p>
      </div>
      <div class="dataset-detail">
        <h3>2. dataset_benchmark_mcq_with_series</h3>
        <p>这是从增强版派生出的选择题数据集，共 <strong>{len(mcq)}</strong> 条样本。每条样本有 4 个选项、唯一 <code>answer_key</code>、正确选项、干扰项分析和 <code>raw_data</code>。</p>
        <p><strong>任务类型：</strong>与生成式增强版一致，共 {len(v2_task)} 类任务。</p>
        <p><strong>答案位置分布：</strong>{esc(dict(mcq_answers))}，整体较均衡，避免模型通过位置偏置获益。</p>
        <p><strong>干扰项特点：</strong>错误选项不是明显荒谬答案，而是常见但次优的方案，例如只做 holdout 但全量拟合 scaler、只看 RMSE 忽略延迟、全局校准但忽略促销切片等。</p>
        <p><strong>适合评测：</strong>自动化、可复现、可计算 accuracy 的模型性能测试，尤其适合比较 text-only 与 image-text 的差异。</p>
      </div>
      <div class="dataset-detail">
        <h3>3. timeseries_agent_benchmark_cases</h3>
        <p>这是完整 case-level benchmark，共 <strong>{case_summary['num_cases']}</strong> 个 case、<strong>{case_summary['num_questions']}</strong> 道题。每个 case 都有完整训练/验证数据，而不是单条序列描述。</p>
        <p><strong>Case 类型：</strong>{esc(dict(case_summary['case_type_counts']))}。</p>
        <p><strong>题型分布：</strong>{esc(dict(case_summary['question_type_counts']))}；输入模式分布为 {esc(dict(case_summary['input_mode_counts']))}。</p>
        <p><strong>每个 case 的文件：</strong><code>train.csv</code>、<code>valid.csv</code>、<code>metadata.json</code>、<code>basic_statistics.json</code>、<code>data_dictionary.md</code>、隐藏 <code>generation_config.json</code>、6 张图、题目文件和 <code>answer_key.json</code>。</p>
        <p><strong>适合评测：</strong>端到端 Agent 能力，包括多文件读取、字段可用性判断、验证方案设计、泄漏检测、残差诊断、多模态图像利用和排序式决策。</p>
      </div>
    </section>
    """


def rubric_sections(v2: list[dict[str, Any]], mcq: list[dict[str, Any]]) -> str:
    v2_sample = v2[0]
    mcq_sample = mcq[0]
    case_scoring = json.loads((CASES / "evaluation" / "scoring_rules.json").read_text(encoding="utf-8"))
    criteria = v2_sample["scoring"]["base_criteria"]
    criteria_rows = "\n".join(
        f"<tr><td>{esc(name)}</td><td class=\"num\">{points}</td><td>{esc(desc)}</td></tr>"
        for name, points, desc in [
            ("数据特征识别", criteria["数据特征识别"], "是否识别趋势、季节性、异常、变点、缺失、间歇性等核心结构。"),
            ("模型或方案合理性", criteria["模型或方案合理性"], "模型/方案是否匹配数据规模、horizon、外生变量、业务场景。"),
            ("验证与指标", criteria["验证与指标"], "是否使用 rolling/expanding backtesting，是否避免泄漏，指标是否适配业务。"),
            ("结果分析深度", criteria["结果分析深度"], "是否能解释误差、残差结构、预测区间和失败模式。"),
            ("可执行性", criteria["可执行性"], "建议是否能落地到数据处理、训练、验证、上线监控流程。"),
            ("风险意识", criteria["风险意识"], "是否意识到漂移、泄漏、异常、成本不对称和不确定性风险。"),
        ]
    )
    return f"""
    <section>
      <h2>评测 Rubric 与评分公式</h2>
      <p>三套数据集都有对应评分方式，但粒度不同：生成式版是 rubric 评分，MCQ 版是客观 exact match，case-level 版同时支持单选、多选、排序题评分。下面给出可直接用于评测脚本或报告的定义。</p>

      <div class="dataset-detail">
        <h3>1. dataset_benchmark_v2_with_series：生成式 Rubric</h3>
        <p>每条样本包含 <code>rubric</code> 和 <code>scoring</code> 字段。基础分为 10 分，图文评测可额外计算 2 分图像利用加分。建议人工评分或用 LLM-as-judge 辅助评分，但需要抽样人工校准。</p>
        <table>
          <thead><tr><th>评分维度</th><th class="num">分值</th><th>判分要点</th></tr></thead>
          <tbody>{criteria_rows}</tbody>
        </table>
        <p><strong>基础总分公式：</strong></p>
        <pre>base_score = feature_identification
           + model_or_plan
           + validation_and_metrics
           + analysis_depth
           + executability
           + risk_awareness

base_score ∈ [0, 10]</pre>
        <p><strong>图文额外分：</strong></p>
        <pre>image_bonus ∈ [0, 2]
final_score_image_text = base_score + image_bonus

image_bonus 评估模型是否真正利用图中的形状、异常、残差结构、区间覆盖等信息，
而不是只复述文本统计。</pre>
        <p><strong>当前样本 rubric 原文：</strong></p>
        <pre>{esc(json.dumps(v2_sample["rubric"], ensure_ascii=False, indent=2))}</pre>
      </div>

      <div class="dataset-detail">
        <h3>2. dataset_benchmark_mcq_with_series：单选 Exact Match</h3>
        <p>每条样本有唯一 <code>answer_key</code>。干扰项是“部分合理但不是最佳”的次优方案，因此适合客观测试模型是否抓住关键约束。</p>
        <pre>{esc(json.dumps(mcq_sample["scoring"], ensure_ascii=False, indent=2))}</pre>
        <p><strong>单题公式：</strong></p>
        <pre>score_i = 1 if predicted_option_i == answer_key_i else 0</pre>
        <p><strong>整体准确率：</strong></p>
        <pre>accuracy = (Σ score_i) / N</pre>
        <p><strong>多模态增益：</strong></p>
        <pre>accuracy_text_only  = correct_text_only / N
accuracy_image_text = correct_image_text / N
multimodal_gain = accuracy_image_text - accuracy_text_only</pre>
        <p>建议同时按 <code>task_type</code>、<code>raw_data.type</code>、<code>difficulty_level</code> 分组报告 accuracy。</p>
      </div>

      <div class="dataset-detail">
        <h3>3. timeseries_agent_benchmark_cases：单选 / 多选 / 排序综合评分</h3>
        <p>评分规则保存在 <code>timeseries_agent_benchmark_cases/evaluation/scoring_rules.json</code>。评测时不要把 <code>answer_key.json</code> 和 <code>generation_config.json</code> 给 Agent。</p>
        <pre>{esc(json.dumps(case_scoring, ensure_ascii=False, indent=2))}</pre>
        <h4>单选题</h4>
        <pre>score_i = 1 if selected == answer[0] else 0</pre>
        <h4>多选题：推荐 F1</h4>
        <pre>precision = |selected ∩ gold| / |selected|
recall    = |selected ∩ gold| / |gold|
F1        = 2 * precision * recall / (precision + recall)

如果 selected 为空，precision 记为 0；如果 precision + recall = 0，F1 = 0。</pre>
        <h4>多选题：可选惩罚式 partial credit</h4>
        <pre>score = max(0,
            correct_hits / num_gold
          - false_positives / num_non_gold)</pre>
        <h4>排序题：Top-2 ordered accuracy</h4>
        <pre>top2_score = 1 if predicted_ranking[:2] == ideal_ranking[:2] else 0</pre>
        <h4>排序题：Kendall tau 归一化</h4>
        <pre>tau = (concordant_pairs - discordant_pairs) / total_pairs
normalized_tau = (tau + 1) / 2</pre>
        <h4>排序题：nDCG</h4>
        <pre>DCG  = Σ relevance_i / log2(position_i + 1)
IDCG = ideal ranking 的 DCG
nDCG = DCG / IDCG</pre>
        <p><strong>case-level 总分建议：</strong>分别计算 text-only 与 image-text 分数，再报告差值。</p>
        <pre>case_score = mean(question_scores_in_case)
overall_score = mean(case_score over all cases)
multimodal_gain = overall_score_image_text - overall_score_text_only</pre>
      </div>
    </section>
    """


def raw_preview(raw_data: dict[str, Any]) -> str:
    if "series" in raw_data:
        preview = raw_data["series"][:3]
        length = len(raw_data["series"])
    elif "series_by_id" in raw_data:
        first_key = next(iter(raw_data["series_by_id"]))
        preview = {first_key: raw_data["series_by_id"][first_key][:8]}
        length = len(raw_data["series_by_id"][first_key])
    elif "models" in raw_data:
        preview = raw_data["models"][:3]
        length = len(raw_data["models"])
    else:
        preview = raw_data
        length = "-"
    return f"<p><strong>raw_data.type:</strong> <code>{esc(raw_data.get('type'))}</code>；<strong>长度:</strong> {esc(length)}</p><pre>{esc(json.dumps(preview, ensure_ascii=False, indent=2))}</pre>"


def sample_v2(records: list[dict[str, Any]]) -> str:
    sample = records[6]
    img = f"dataset_benchmark_v2_with_series/{sample['input']['multimodal']['image']}"
    return f"""
    <article class="sample">
      <div><img src="{esc(img)}" alt="v2 sample"></div>
      <div>
        <div class="eyebrow">{esc(sample['id'])} · {esc(sample['task_type'])}</div>
        <h3>生成式问答样例</h3>
        <p>{esc(truncate(sample['input']['text_only']['question']))}</p>
        <h4>参考答案摘录</h4>
        <pre>{esc(json.dumps(sample['answer'], ensure_ascii=False, indent=2)[:900])}</pre>
        <h4>原始序列片段</h4>
        {raw_preview(sample['raw_data'])}
      </div>
    </article>
    """


def sample_mcq(records: list[dict[str, Any]]) -> str:
    sample = records[10]
    img = f"dataset_benchmark_mcq_with_series/{sample['input']['multimodal']['image']}"
    choices = "\n".join(f"{c['id']}. {c['text']}" for c in sample["choices"])
    return f"""
    <article class="sample">
      <div><img src="{esc(img)}" alt="mcq sample"></div>
      <div>
        <div class="eyebrow">{esc(sample['id'])} · {esc(sample['task_type'])}</div>
        <h3>选择题样例</h3>
        <p>{esc(truncate(sample['input']['text_only']['question'], 900))}</p>
        <h4>选项</h4>
        <pre>{esc(choices)}</pre>
        <p><strong>answer_key:</strong> <code>{esc(sample['answer_key'])}</code></p>
        <h4>干扰项分析</h4>
        <pre>{esc(json.dumps(sample['distractor_analysis'], ensure_ascii=False, indent=2))}</pre>
        <h4>原始序列/结构化数据片段</h4>
        {raw_preview(sample['raw_data'])}
      </div>
    </article>
    """


def sample_case() -> str:
    case_dir = CASES / "cases" / "case_026"
    meta = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = json.loads((case_dir / "generation_config.json").read_text(encoding="utf-8"))
    text_qs = json.loads((case_dir / "questions" / "text_only_questions.json").read_text(encoding="utf-8"))
    img = "timeseries_agent_benchmark_cases/cases/case_026/figures/overview.png"
    rows = (case_dir / "train.csv").read_text(encoding="utf-8").splitlines()[:5]
    q = text_qs[3]
    return f"""
    <article class="sample">
      <div><img src="{esc(img)}" alt="case overview"></div>
      <div>
        <div class="eyebrow">case_026 · {esc(meta['case_type_name'])}</div>
        <h3>完整 Case 样例</h3>
        <p><strong>可见 metadata:</strong> 频率 {esc(meta['frequency'])}，序列数 {esc(meta['num_series'])}，预测 horizon {esc(meta['forecast_horizon'])}</p>
        <p><strong>隐藏标签摘录:</strong> leakage_columns = <code>{esc(cfg.get('leakage_columns'))}</code></p>
        <h4>train.csv 前几行</h4>
        <pre>{esc(chr(10).join(rows))}</pre>
        <h4>题目样例</h4>
        <pre>{esc(json.dumps(q, ensure_ascii=False, indent=2))}</pre>
      </div>
    </article>
    """


def build() -> str:
    v2 = read_jsonl(V2 / "ts_agent_qa.jsonl")
    mcq = read_jsonl(MCQ / "ts_agent_mcq.jsonl")
    case_summary = json.loads((CASES / "summary.json").read_text(encoding="utf-8"))
    v2_task = Counter(r["task_type"] for r in v2)
    raw_types = Counter(r["raw_data"]["type"] for r in v2)
    mcq_answers = Counter(r["answer_key"] for r in mcq)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>时序 Agent Benchmark 数据集使用指南</title>
  <style>
    :root {{ --ink:#17202a; --muted:#5d6978; --line:#d9e2ec; --soft:#f6f8fb; --blue:#2563eb; --green:#0f766e; --amber:#b45309; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; color:var(--ink); background:#edf2f7; line-height:1.58; }}
    header {{ background:linear-gradient(135deg,#14213d,#1f5f68 60%,#386641); color:white; padding:54px max(22px,6vw) 38px; }}
    header h1 {{ margin:0 0 12px; font-size:clamp(30px,5vw,56px); line-height:1.06; }}
    header p {{ max-width:1040px; margin:0; color:#e7eef7; font-size:18px; }}
    main {{ padding:28px max(18px,5vw) 70px; }}
    section {{ max-width:1320px; margin:0 auto 26px; background:white; border:1px solid var(--line); border-radius:8px; padding:24px; box-shadow:0 10px 28px rgba(18,35,56,.06); }}
    h2 {{ margin:0 0 16px; font-size:25px; }}
    h3 {{ margin:0 0 10px; font-size:19px; }}
    h4 {{ margin:14px 0 6px; font-size:15px; }}
    code {{ background:#edf2f7; padding:1px 5px; border-radius:4px; }}
    .kpis {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; }}
    .kpi {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:15px; }}
    .kpi strong {{ display:block; font-size:30px; line-height:1; margin-bottom:7px; }}
    .grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; }}
    .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }}
    .card {{ background:var(--soft); border:1px solid var(--line); border-radius:8px; padding:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th,td {{ border-bottom:1px solid var(--line); padding:8px; text-align:left; vertical-align:middle; }}
    th {{ background:#f8fafc; color:var(--muted); }}
    .compare-table th:first-child {{ width:120px; color:var(--ink); }}
    .compare-table td {{ width:29%; }}
    .num {{ text-align:right; width:64px; font-variant-numeric:tabular-nums; }}
    .bar {{ height:10px; background:#e7edf4; border-radius:999px; overflow:hidden; min-width:90px; }}
    .bar span {{ display:block; height:100%; background:var(--blue); }}
    .sample {{ display:grid; grid-template-columns:minmax(320px,42%) 1fr; gap:20px; border-top:1px solid var(--line); padding-top:20px; }}
    .sample img {{ width:100%; display:block; border:1px solid var(--line); border-radius:8px; background:white; }}
    .eyebrow {{ color:var(--muted); font-size:13px; }}
    pre {{ white-space:pre-wrap; word-break:break-word; background:#0f172a; color:#e5edf8; padding:12px; border-radius:8px; font-size:12px; overflow:auto; }}
    .callout {{ background:#fff8ed; border-left:4px solid var(--amber); padding:12px 14px; border-radius:6px; }}
    .good {{ background:#eefdf7; border-left:4px solid var(--green); padding:12px 14px; border-radius:6px; }}
    .dataset-detail {{ border:1px solid var(--line); border-radius:8px; padding:16px; margin:14px 0; background:#fbfdff; }}
    .dataset-detail p {{ margin:8px 0; }}
    ul {{ padding-left:20px; }}
    @media (max-width: 920px) {{ .kpis,.grid2,.grid3,.sample {{ grid-template-columns:1fr; }} section {{ padding:18px; }} }}
  </style>
</head>
<body>
<header>
  <h1>时序建模 Agent Benchmark 数据集使用指南</h1>
  <p>本文介绍三套当前推荐使用的数据集：生成式问答版、选择题版、完整 case-level 版，并说明如何构造 text-only 与 image-text 两种评测输入，以及如何计算分数。</p>
</header>
<main>
  <section>
    <h2>三套数据集定位</h2>
    <div class="grid3">
      <div class="card">
        <h3>dataset_benchmark_v2_with_series</h3>
        <p>生成式问答数据。每条样本包含原始序列 <code>raw_data</code>、文本问题、图片路径、参考答案和评分 rubric。</p>
        <p><strong>适合：</strong>人工质检、分析 Agent 推理质量、构造开放任务。</p>
      </div>
      <div class="card">
        <h3>dataset_benchmark_mcq_with_series</h3>
        <p>选择题数据。继承增强版的 <code>raw_data</code> 和图像，并加入 <code>choices</code>、<code>answer_key</code>、干扰项分析。</p>
        <p><strong>适合：</strong>自动化客观评测、text-only vs image-text 对比。</p>
      </div>
      <div class="card">
        <h3>timeseries_agent_benchmark_cases</h3>
        <p>完整建模场景数据。每个 case 包含 <code>train.csv</code>、<code>valid.csv</code>、metadata、隐藏配置、6 类图像和 paired questions。</p>
        <p><strong>适合：</strong>项目式 Agent benchmark，测试文件读取、数据理解、验证设计和多模态诊断。</p>
      </div>
    </div>
  </section>

  <section>
    <h2>三套数据集横向对比</h2>
    <p>下面这张表从定位、规模、题型、输入形态、答案形式和推荐用途等维度对比三套数据集。简单说：生成式版适合分析推理质量，MCQ 版适合自动化打分，case-level 版适合端到端 Agent 项目式评测。</p>
    {dataset_comparison_table(v2, mcq, case_summary)}
  </section>

  {detail_intro_sections(v2, mcq, case_summary)}

  <section>
    <h2>规模总览</h2>
    <div class="kpis">
      <div class="kpi"><strong>{len(v2)}</strong><span>生成式问答样本</span></div>
      <div class="kpi"><strong>{len(mcq)}</strong><span>选择题样本</span></div>
      <div class="kpi"><strong>{case_summary['num_cases']}</strong><span>完整 cases</span></div>
      <div class="kpi"><strong>{case_summary['num_questions']}</strong><span>case-level 题目</span></div>
    </div>
  </section>

  <section>
    <h2>dataset_benchmark_v2_with_series</h2>
    <div class="grid2">
      <div>
        <h3>任务分布</h3>
        {counter_table(v2_task, "任务类型")}
      </div>
      <div>
        <h3>raw_data 类型</h3>
        {counter_table(raw_types, "raw_data.type")}
      </div>
    </div>
    <div class="good">
      <strong>评测方式：</strong>这套更适合生成式评测。text-only 输入应包含 <code>raw_data</code>、<code>input.text_only.question</code>、业务字段和必要 metadata；image-text 输入在同样内容上额外提供 <code>input.multimodal.image</code>。评分可用 rubric 人工评分，或用 LLM-as-judge 但需抽样人工校验。
    </div>
    {sample_v2(v2)}
  </section>

  <section>
    <h2>dataset_benchmark_mcq_with_series</h2>
    <div class="grid2">
      <div>
        <h3>任务分布</h3>
        {counter_table(Counter(r['task_type'] for r in mcq), "任务类型")}
      </div>
      <div>
        <h3>答案位置分布</h3>
        {counter_table(mcq_answers, "answer_key")}
      </div>
    </div>
    <div class="good">
      <strong>推荐评测方式：</strong>每题单选 exact match。text-only 输入提供 <code>raw_data + input.text_only.question + choices</code>，不提供图片；image-text 输入提供同样内容并额外提供图片。分别计算 accuracy，再报告 <code>multimodal_gain = accuracy_image_text - accuracy_text_only</code>。
    </div>
    {sample_mcq(mcq)}
  </section>

  <section>
    <h2>timeseries_agent_benchmark_cases</h2>
    <div class="grid2">
      <div>
        <h3>Case 类型</h3>
        {counter_table(Counter(case_summary['case_type_counts']), "Case 类型")}
      </div>
      <div>
        <h3>题型与输入模式</h3>
        {counter_table(Counter(case_summary['question_type_counts']), "题型")}
        <br>
        {counter_table(Counter(case_summary['input_mode_counts']), "输入模式")}
      </div>
    </div>
    <div class="callout">
      <strong>目录结构：</strong>每个 case 是一个完整建模场景，包含 <code>train.csv</code>、<code>valid.csv</code>、<code>metadata.json</code>、<code>data_dictionary.md</code>、<code>basic_statistics.json</code>、隐藏 <code>generation_config.json</code>、6 张图、题目和答案。评测时不要把 <code>generation_config.json</code> 和 <code>answer_key.json</code> 给 Agent。
    </div>
    {sample_case()}
  </section>

  {rubric_sections(v2, mcq)}

  <section>
    <h2>具体评测协议</h2>
    <div class="grid2">
      <div class="card">
        <h3>1. JSONL 选择题评测</h3>
        <ul>
          <li>数据：<code>dataset_benchmark_mcq_with_series/ts_agent_mcq.jsonl</code></li>
          <li>text-only：只给 <code>raw_data</code>、文本问题和 <code>choices</code></li>
          <li>image-text：额外给 <code>input.multimodal.image</code></li>
          <li>输出格式：要求模型只输出 A/B/C/D</li>
          <li>评分：<code>accuracy = correct / total</code></li>
          <li>报告：按任务类型、raw_data 类型、难度层级分别统计</li>
        </ul>
      </div>
      <div class="card">
        <h3>2. Case-level 评测</h3>
        <ul>
          <li>数据：<code>timeseries_agent_benchmark_cases/questions/*.jsonl</code></li>
          <li>text-only：按 <code>visible_files</code> 提供 CSV、metadata、字典和统计摘要</li>
          <li>image-text：在同一 case 上额外提供 <code>figures/</code> 图像</li>
          <li>单选：exact match</li>
          <li>多选：推荐 F1(selected, gold)</li>
          <li>排序：Kendall tau、nDCG 或 top-2 ordered accuracy</li>
        </ul>
      </div>
    </div>
  </section>

  <section>
    <h2>评分细则</h2>
    <div class="grid3">
      <div class="card">
        <h3>单选题</h3>
        <p>模型输出与标准答案完全一致得 1，否则得 0。</p>
        <pre>score = 1 if pred == answer_key else 0</pre>
      </div>
      <div class="card">
        <h3>多选题</h3>
        <p>推荐用 F1，既惩罚漏选也惩罚多选。</p>
        <pre>precision = |pred ∩ gold| / |pred|
recall = |pred ∩ gold| / |gold|
F1 = 2PR / (P + R)</pre>
      </div>
      <div class="card">
        <h3>排序题</h3>
        <p>简单报告可用 top-2 ordered accuracy；完整报告可加 Kendall tau / nDCG。</p>
        <pre>top2 = 1 if pred[:2] == gold[:2] else 0
multimodal_gain = score_image - score_text</pre>
      </div>
    </div>
  </section>

  <section>
    <h2>推荐报告指标</h2>
    <ul>
      <li><strong>总体分数：</strong>text-only 与 image-text 分别计算。</li>
      <li><strong>多模态增益：</strong><code>image_text_score - text_only_score</code>。</li>
      <li><strong>按任务切片：</strong>模型推荐、泄漏检测、残差诊断、区间预测、层级预测等。</li>
      <li><strong>按数据形态切片：</strong>univariate、multivariate、prediction_interval、hierarchical_series 等。</li>
      <li><strong>错误分析：</strong>利用 <code>distractor_analysis</code> 或 answer explanation 统计常见误区。</li>
    </ul>
  </section>
</main>
</body>
</html>
"""


def main() -> None:
    OUT.write_text(build(), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
