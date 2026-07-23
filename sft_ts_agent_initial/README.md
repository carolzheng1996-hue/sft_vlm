# VLM 时序 Agent SFT 初版数据生成

这个目录用于把现有 `timeseries_agent_benchmark_cases` 转成更适合 SFT 的“工具选择 + 工具调用顺序 + 专家答案”样本。

目标能力：

- 数据画像：趋势、异常点、频率、平稳性、季节性、缺失、异常、趋势识别。
- 相似性分析：共同模式、趋势一致性、方差/分布相似性、聚类或距离选择。
- 模型结果分析：预测残差、训练/验证曲线、漂移检测、重训练触发。
- 模型选择：按数据模式、业务目标、样本规模、频率、可用协变量和评估指标选择模型。
- 模态路由：让模型学会什么时候必须看图，什么时候 text-only 足够，什么时候先统计再看图。

## 文件

- `question_blueprints.json`：初版问题模板，覆盖序列级、数据集级、工具路由和模型选择。
- `tool_model_catalog.template.json`：备用工具清单和模型清单模板。
- `scripts/build_seed_questions.py`：读取已有 case，生成待回答的 SFT seed questions。
- `scripts/generate_answers.py`：调用 OpenAI-compatible Chat Completions API，为 seed questions 生成专家答案；支持同模型多次生成和择优融合。
- `scripts/review_answers.py`：用确定性规则和多个 judge 模型评审生成结果。
- `scripts/recycle_low_quality.py`：把低分、错误或过简单样本回炉成二次 seed questions。

## 快速开始

生成初版问题：

```bash
python sft_ts_agent_initial/scripts/build_seed_questions.py \
  --cases-root timeseries_agent_benchmark_cases/cases \
  --catalog sft_ts_agent_initial/tool_model_catalog.template.json \
  --model-metadata metadata_1.json metadata_2.json metadata_3.json \
  --tools-source export_data_analysis_tools.py \
  --blueprints sft_ts_agent_initial/question_blueprints.json \
  --out sft_ts_agent_initial/output/seed_questions.jsonl \
  --max-cases 40
```

如果你已经用 `export_data_analysis_tools.py` 在完整 TSA 仓库中导出了工具 bundle，也可以用更精确的工具定义：

```bash
python sft_ts_agent_initial/scripts/build_seed_questions.py \
  --cases-root timeseries_agent_benchmark_cases/cases \
  --catalog sft_ts_agent_initial/tool_model_catalog.template.json \
  --model-metadata metadata_1.json metadata_2.json metadata_3.json \
  --tools-bundle tool_dataset_sources/data_analysis_tools_bundle.json \
  --blueprints sft_ts_agent_initial/question_blueprints.json \
  --out sft_ts_agent_initial/output/seed_questions.jsonl \
  --max-cases 40
```

调用大模型生成答案。建议同一个模型采样 3 次，再让同模型做一次择优融合：

```bash
export OPENAI_API_KEY="你的 key"
python sft_ts_agent_initial/scripts/generate_answers.py \
  --input sft_ts_agent_initial/output/seed_questions.jsonl \
  --output sft_ts_agent_initial/output/sft_messages.jsonl \
  --model gpt-4.1 \
  --base-url https://api.openai.com/v1 \
  --num-samples 3 \
  --sample-temperature 0.4 \
  --select-best
```

评审生成结果。没有 judge 模型时也会跑确定性规则；有多个 judge 模型时用逗号分隔：

```bash
python sft_ts_agent_initial/scripts/review_answers.py \
  --input sft_ts_agent_initial/output/sft_messages.jsonl \
  --output sft_ts_agent_initial/output/reviews.jsonl \
  --judge-models gpt-4.1,o3
```

把低分、错误或过简单样本回炉：

```bash
python sft_ts_agent_initial/scripts/recycle_low_quality.py \
  --seeds sft_ts_agent_initial/output/seed_questions.jsonl \
  --reviews sft_ts_agent_initial/output/reviews.jsonl \
  --out sft_ts_agent_initial/output/recycled_seed_questions.jsonl
```

然后对 `recycled_seed_questions.jsonl` 再跑一次 `generate_answers.py` 和 `review_answers.py`。

## 推荐生产流程

1. 生成 seed：覆盖 text-only、image-text、tool-trace、model-selection，不要只做问答解释。
2. 多样本生成：同一强模型生成 2-5 次，保留候选，融合出最终答案。
3. 规则评审：先查 schema、工具覆盖、图像 grounding、答案长度、复杂度。
4. 多模型评审：至少 2 个 judge，避免单个模型偏好污染数据。
5. 回炉再造：`reply_error` 修答案，`question_too_simple` 升级问题，`needs_rewrite` 同时改问题和答案。
6. 过滤合并：只把 `pass` 或二次评审达标样本进入训练集；保留失败样本用于 hard negative 或评测，不直接 SFT。

## 质量门槛建议

- `image_text` 样本必须明确“哪些判断来自图像”，否则容易训练成装作看图。
- 工具调用答案必须有顺序原因，不能只罗列工具名。
- 模型选择答案必须包含 baseline、候选模型、验证设计、指标和不选理由。
- 对隐藏生成配置可知但用户不可见的事实，答案应表达为“根据当前可见信息/需要工具确认”，避免训练模型臆造。
- 每个 case 不要只生成同构问题。建议按 case 类型控制采样：单序列重画像，多序列重相似性/全局模型，残差 case 重诊断/重训练。
- 留出一部分低质量和过简单样本作为评审器回归测试集，不要全部覆盖掉。

输出是 JSONL。每行包含：

- `messages`：可直接改造成 chat SFT 格式。
- `images`：该样本推荐提供给 VLM 的图片路径。
- `metadata`：case、任务类型、候选工具、候选模型、可见文件、答案生成状态。
- `expert_trace`：期望答案中出现的工具先后关系和决策要点，供质检或二次过滤使用。

## 推荐数据配比

初版可按下面比例采样：

- 数据画像/模式识别：30%
- 相似性分析：15%
- 模型结果分析：20%
- 模型选择：25%
- 端到端开放任务：10%

每类任务建议同时保留：

- `text_only`：模型只基于 schema、统计摘要和少量样本回答。
- `image_text`：模型必须结合图像诊断。
- `tool_trace`：模型先声明工具调用计划，再基于工具结果给结论。
