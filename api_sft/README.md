# API VLM 时序 SFT 数据构建

该目录生成两类高难度 VLM SFT Question：只提供结构化文本的 `text_only` 样本，以及真实附带时序图像的 `image_text` 样本。Question 采用“确定性 QuestionSpec + LLM 直接写题”，避免把标准解题步骤直接泄露给模型。

Answer 阶段采用真实闭环：OpenAI-compatible 模型逐轮产生 `tool_calls`，程序在隔离 session 中执行现有 `claude_tsa` handler，将真实结果回填后再请求下一步，直到形成最终回答。

## 安装与配置

```bash
python -m pip install -r api_sft/requirements.txt
cp api_sft/config.example.yaml api_sft/config.yaml
```

编辑 `config.yaml` 中各模型的 `base_url` 和 `model`。API key 支持以下两种配置方式。

合成场景默认使用 V4 组件生成器；面板数据在模块边界统一使用 `wide_panel_v1`：

```yaml
scenario_generation:
  version: v4
  output_subdir: scenarios
  complexity_mix:
    controlled: 0.20
    compositional: 0.60
    confounded: 0.20
  oracle_free_images: true
  multiseries:
    quality_gate: true
    max_attempts: 12
    general_correlation_mix: {low: 0.25, medium: 0.50, high: 0.25}
```

`controlled` 保留可解释的单一模式，`compositional` 组合趋势、多个季节项、相关噪声、协变量和局部事件，`confounded` 则加入周期漂移、局部异方差、条件缺失或部分序列受影响等混杂因素。V4 多序列由公共、组级和个体因子共同生成，并按模板检查原始、差分、去趋势、组内和组间相关；不合格样本最多确定性重采样 12 次。主值列使用 `s01,s02,...`，其他指标使用 `s01__prediction`、`s01__residual` 等名称。重复观测保留为稀疏重复 time 行，不聚合、不暴露 occurrence 辅助列。

推荐方式是复制 `.env.example`：

```bash
cp api_sft/.env.example api_sft/.env
```

在 `api_sft/.env` 中填写：

```dotenv
OPENAI_API_KEY="..."
SECOND_VLM_API_KEY="..."
```

并在 `config.yaml` 顶层设置：

```yaml
env_file: api_sft/.env
env_override: false
```

各模型使用变量名引用密钥：

```yaml
api_key_env: OPENAI_API_KEY
api_key: null
```

也可以不使用 `.env`，在启动命令的同一个终端设置环境变量：

```bash
export OPENAI_API_KEY="..."
export SECOND_VLM_API_KEY="..."
```

如果明确希望将密钥直接写入本地 `config.yaml`，使用：

```yaml
api_key_env: null
api_key: "..."
```

当两种方式同时配置时，`api_key_env` 对应的环境变量优先，`api_key` 作为回退。`config.yaml` 和 `.env` 都已被 `.gitignore` 排除；不要把真实密钥写入 `config.example.yaml` 或 `.env.example`。

默认工具源码真源指向：

`/Users/monychen/Documents/tsa/claude_tsa`

`prepare-catalogs` 会从当前源码冻结 `data_analysis_tools_bundle.json`，轨迹执行时则从 live `ToolRegistry` 读取完整 `input_schema`。由于当前 `claude_tsa` 使用 `datetime.UTC`，轨迹命令需要 Python 3.11+；本机可使用 `claude_tsa_qa_py312` 环境。

## 分阶段运行

```bash
python -m api_sft --config api_sft/config.yaml prepare-catalogs
python -m api_sft --config api_sft/config.yaml migrate-scenarios --resume
python -m api_sft --config api_sft/config.yaml generate-question-specs
python -m api_sft --config api_sft/config.yaml generate-questions --fresh
```

`generate-question-specs` 不调用 API；`generate-questions` 使用 `models.question`。首次生成或协议升级使用 `--fresh`，已有 final、audit、rejected 和 coverage 会被移动到 `question_run_archives/<timestamp>/`；只有同一 Final/Audit 协议的中断续跑才使用 `--resume`。快速小规模检查可以给两个命令加 `--limit`。`rewrite-questions` 暂时保留为 `generate-questions` 的弃用别名。

Question Writer 会按问题组稳定分配目标、约束、不确定性、材料、权衡、复盘、交接或条件优先的表达风格；text/image 配对继续共享同一问题。Writer 只接收精简的外部业务上下文，不接收数据规模、统计摘要或观测结论。生成阶段会拒绝跨组重复、相似度高于 `0.93` 的近重复、高频相同开场，以及未经提供的数量、频率和数据结论；风格与去重信息记录在 `questions.audit.jsonl` 和 coverage 报告中。

### 独立并发 Question 生成器

如需加速 Question 生成，可使用独立的 `questions_concurrent` 模块。它保持与串行 `generate-questions` 相同的 QuestionSpec、提示词、质量校验、去重和输出协议；每个 `question_group` 单独占用一个并发槽，不会把多个不同问题合并到一次 LLM 请求中。paired group 仍只调用一次 LLM，并物化为共享相同问题文本的 `text-only` 和 `image-text` 两条记录。

当前数据中的 `question_specs.jsonl` 包含 506 个 `question_group`、622 条 spec。生成全部 group（不要传 `--limit`）可执行：

```bash
python -m api_sft.questions_concurrent \
  --config api_sft/config.yaml \
  --max-concurrency 4 \
  --fresh
```

并发结果写入独立目录 `output/concurrent_questions/`，不会覆盖串行生成器的 `questions.final.jsonl`、audit、rejected 或 coverage。`--fresh` 会先归档该并发子目录中的旧产物；如果任务中断，可用以下命令续跑：

```bash
python -m api_sft.questions_concurrent \
  --config api_sft/config.yaml \
  --max-concurrency 4 \
  --resume
```

`--max-concurrency` 命令行参数优先于 `generation.max_concurrency`（当前默认值为 4）；`--limit N` 仅用于小规模试跑。以当前模型和历史请求延迟估算，506 个 group 在并发 4 下首次通过通常约需 55--65 分钟，考虑校验重试后建议预留 1--1.5 小时，遇到限流或大量重试时可能接近 2 小时。paired group 虽然最终产生两条记录，但只计一次 API 请求，因此最终记录数通常接近 622 条，实际数量以 rejected 和 coverage 报告为准。

没有旧场景时可改用 `generate-scenarios` 直接生成当前场景。迁移命令默认读取 `output/legacy/scenarios_v3/scenarios.jsonl`；旧版 Question 不会跨协议复用，记录内部仍通过 `generator_version`、`question_spec_version` 和 `question_contract_version` 标识协议。

完整执行：

```bash
python -m api_sft --config api_sft/config.yaml run-all --resume
```

`run-all` 运行到 Question 生成后停止；然后运行真实轨迹阶段：

```bash
/Users/monychen/miniforge3/envs/claude_tsa_qa_py312/bin/python -m api_sft \
  --config api_sft/config.yaml run-trajectories --fresh --limit 20
```

首次试跑或格式升级后使用 `--fresh`：已有轨迹会被移动到 `trajectory_run_archives/<timestamp>/`，不会直接删除。只有同一轨迹格式的中断续跑使用 `--resume`；两个参数互斥。

也可分别执行 `generate-trajectories`、`verify-trajectories` 和 `export-trajectories`。旧的文字工具计划 `generate-answers` 仍被保护性禁用，避免与真实轨迹混用。

## 数据设计

- `coverage_matrix.yaml` 保存目标覆盖矩阵。
- `task_pool.yaml` 保存 58 个细粒度任务，覆盖五类任务和全部 33 个工具。
- [QUESTION_CONSTRUCTION.md](QUESTION_CONSTRUCTION.md) 详细说明场景调度、任务匹配、配对模态、复杂度校验和覆盖验证流程。
- 每个场景保存 CSV、可见摘要、PNG、图片 provenance、隐藏真值、随机种子、组件 hash 和场景 hash。
- V4 信号由趋势、0–3 个季节项、协变量作用和多类噪声组合生成；周期可为非整数并支持幅度调制、周期漂移和季节结构突变。
- 多序列真实包含全局公共因子、组级因子和序列独有因子；普通 panel 分层覆盖低、中、高相关，特定相似性模板使用独立目标区间。
- `signal_complexity`、`analysis_difficulty` 和 `image_value` 相互独立；图片价值高不再自动意味着问题困难。
- 普通诊断图片只读取观测数据。STL 周期、异常和变点均由确定性诊断算法估计，允许与隐藏真值不一致；`figure_provenance.json` 对每张图记录输入、方法、参数及 `oracle_truth_used=false`。
- `--resume` 只复用 V4、质量门槛通过、文件 hash 有效且 `component_hash` 一致的完整场景。
- `generate-question-specs` 会拒绝旧生成器场景；升级后必须先运行一次 `generate-scenarios`。
- 使用分层模态采样：低图像价值以 text-only 为主，中价值提高严格配对比例，高价值以 image-text 为主；仅部分场景生成严格配对问题。
- 配对记录通过 `pair_id` 和 `split_group` 绑定，划分训练/验证集时必须整组分配。
- 通用工具执行、证据边界和粗粒度模型目录规则放在 `prompts/tool_execution_system.txt`，不再重复写入用户问题。
- 最终 `questions.final.jsonl` 使用 `question_runtime_v2` 紧凑协议，只保留任务、System Prompt、自然请求、资源和工具白名单；System Prompt 在生成时直接读取 `prompts/tool_execution_system.txt` 并写入 `prompt.system_prompt`。
- 轨迹运行时才把暂存后的 `uploads/dataset.csv` URI 加入 user content；final 不保存 `context_block`、重复 `messages` 或完整工具 schema。
- Question Writer 只看到用户目标、外部业务约束和资源语义，不看到行数、历史长短、统计摘要或数据模式；Question Contract v3 同时阻止派生结论和越界模型选型。
- 用户问题包含至少两个决策点和一个现实约束，但不包含固定工具链、Top-3 或答案章节。
- Question LLM、两个 Answer 候选模型和 Selector 都看不到隐藏真值。
- Selector 不执行隐藏真值修订，避免把不可见信息写入训练答案。

## 主要输出

默认位于 `api_sft/output/`：

- `scenarios/scenarios.jsonl`
- `question_specs.jsonl`
- `questions.final.jsonl`
- `questions.audit.jsonl`
- `questions.rejected.jsonl`
- `coverage_report.json`
- `trajectories.raw.jsonl`
- `trajectories.competition_audit.jsonl`
- `trajectories.verified.jsonl`
- `trajectories.rejected.jsonl`
- `trajectory_exports/trajectories.compact.jsonl`
- `trajectory_exports/train_trl_tool_messages.jsonl`

磁盘上的当前文件统一使用无版本后缀，便于人工检查；`v2`、`v4` 只作为 JSON 记录内部的协议/生成器版本，不再出现在当前输出路径中。`trajectory_run_archives/` 中的历史归档保留生成当时的原始文件名，不参与当前续跑。

`questions.final.jsonl` 只保存轨迹运行必需字段；问题生成模型、尝试次数、耗时、token usage、决策点、约束引用和质量检查保存在 `questions.audit.jsonl`。完整证据包和内部 rubric 只存在于 `question_specs.jsonl`，不会进入模型消息。

## 真实 Agent 轨迹

轨迹生成器只把观测 CSV 复制到独立 session 的 `uploads/dataset.csv`，不会预制 channel。模型若需要 channel，必须真实执行 `data_profile → data_convert(target_columns=[...]) → 分析工具`。每题可见工具由任务候选、四个通用数据工具和确定性安全干扰项组成，总数最多 20；System Prompt 不显示 `primary_tools`。

细粒度规则路由已删除。QuestionSpec 只给出 `forecast`、`anomaly_detection` 或 `none` 粗粒度目录范围；Answer 根据真实数据画像按需调用只读 `model_catalog_search`，具体目录模型名只能来自本轮查询结果，Transformer、ARIMA 等通用方法词不按具体目录模型处理。最低成功工具数为 1；schema、session 和可捕获 handler 失败会回填必填字段、允许字段和错误路径供模型修正。

轨迹最多调用 8 次工具。第 7 次结果会提示仅余一次调用，第 8 次后强制生成最终回答。相同错误参数连续失败两次才终止；失败候选在 competition audit 中只保留精简工具时间线、错误和核心指标，不再递归复制完整 QuestionRecord、工具 schema、源码 Git 和 workspace 信息。

每题由 `models.trajectory_candidates` 中恰好两个支持 tool calls 的模型在独立 session 中并行生成。确定性硬校验先淘汰非法轨迹；两条均通过时，`models.trajectory_selector` 接收问题图片和工具生成图片并裁决。competition audit 保存两条候选的人类可读工具时间线、最终回答、评分和失败重试摘要，只有赢家进入验证和 TRL 导出。

当前 Compact 轨迹输出不保存 `git_dirty`、`repo_root`、`workspace_path`、完整 QuestionRecord、逐轮 provider 指标或可推导计数的重复副本。记录内部仍保留格式标识，字段定义见 [TRAJECTORY_COMPACT_SCHEMA.md](TRAJECTORY_COMPACT_SCHEMA.md)。旧版完整实现仍保留在 `trajectories.py`、`trajectory_verify.py` 和 `trajectory_exporters.py`，CLI 使用新的 `*_compact.py` 模块。

`trajectories.compact.jsonl` 是通过验证的人工检查副本。TRL 导出使用 conversational language-modeling 数据：顶层为 `messages`、`tools` 和 `images`，工具参数转换成 Transformers 需要的 Python/JSON object，而不是 OpenAI wire format 的 JSON 字符串；工具结果使用 `role=tool`、`name` 和字符串 `content`。初始图片和工具生成图片转换成 TRL VLM 的 `image` / `text` content blocks。

## 使用 TRL 训练

当前 TRL 官方工具调用格式要求每条样本包含工具调用/结果的 `messages` 以及顶层 `tools`。本地 JSONL 可用 `datasets>=4.7.0` 的 `Json` feature 保留不同工具的任意参数结构；图片路径用 `Image` feature 解码：

```python
import json
from datasets import Dataset, Features, Image, Json, List
from trl import SFTConfig, SFTTrainer

path = "api_sft/output/trajectory_exports/train_trl_tool_messages.jsonl"
with open(path, encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

features = Features({
    "messages": List(Json()),
    "tools": List(Json()),
    "images": List(Image()),
})
dataset = Dataset.from_list(rows, features=features)

trainer = SFTTrainer(
    model="your-tool-capable-model",
    train_dataset=dataset,
    args=SFTConfig(max_length=None),
)
trainer.train()
```

训练图文和纯文本混合数据需要 `transformers>=4.57.0`。目标模型或 processor 的 chat template 必须同时支持工具调用和对应的视觉 content blocks；VLM 建议保持 `max_length=None`，避免截断图片 token。只有当 chat template 支持 assistant token mask 时才开启 `assistant_only_loss=True`。

## 测试

```bash
python -m unittest discover -s api_sft/tests -v
```

测试只使用本地数据和 mock API，不调用真实付费端点。
