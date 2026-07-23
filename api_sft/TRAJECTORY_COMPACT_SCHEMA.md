# Compact 轨迹字段协议

CLI 当前使用以下独立模块，旧版完整轨迹代码保留不动：

- `trajectories_compact.py`
- `trajectory_verify_compact.py`
- `trajectory_exporters_compact.py`

格式版本：

- raw / verified / rejected：`tool_trajectory_v2.2_compact`
- competition audit：`trajectory_competition_audit_v2.1_compact`

这些是记录内部的协议标识，不是当前文件名。当前输出统一使用
`trajectories.raw.jsonl`、`trajectories.verified.jsonl` 和
`trajectories.competition_audit.jsonl`；历史归档才保留旧文件名。

旧格式不能与 compact 格式混合 resume。首次使用 compact 版本应执行 `run-trajectories --fresh`，旧文件会进入 `trajectory_run_archives/<timestamp>/`。

## Raw 与 Verified

| 字段 | 含义 | 是否进入训练 |
|---|---|---|
| `id` | Question ID | 否 |
| `format_version` | Compact 协议版本 | 否 |
| `question` | 仅保留问题文本、任务、输入模态和模型目录大类 | 否 |
| `generation_status` | `ok` 或 `error` | 否 |
| `model` | 胜出候选模型 | 否 |
| `candidate_index` | 胜出候选在配置中的 0/1 序号 | 否 |
| `messages` | 完整 system/user/assistant/tool 轨迹 | 是 |
| `tools` | 本题真实工具定义和 input schema | 是 |
| `tool_calls` | 供人工检查的精简工具时间线 | 否 |
| `final_answer` | 最终回答的便捷索引 | 否，训练使用 messages 中的回答 |
| `metrics` | 调用数、成功数、token 和耗时 | 否 |
| `competition` | 选择规则、胜出候选和候选评分摘要 | 否 |
| `verification` | 仅 verified/rejected 存在；最终确定性检查 | 否 |

`tool_calls` 每项只保留：调用 ID、工具名、参数、成功状态、Answer 的简短决策、结构化结果摘要、错误、artifact URI、图片文件名和目录查询返回的模型名。长数组、深层对象和长字符串会有界截断。

## Competition Audit

顶层只包含：

```text
id
format_version
question
candidates
selection
```

每个 candidate 只包含：

```text
candidate_index
model
generation_status
tool_calls
final_answer
metrics
review
failed_attempts
error（仅失败时）
```

失败 attempt 不再嵌套 QuestionRecord、工具 schema、Git provenance、session workspace 或完整 candidate 副本。

## 明确删除的字段

Compact 输出不保存以下字段：

```text
question_record
tool_source
git_dirty
git_head
repo_root
execution_setup
workspace_path
model_turns
prior_attempt_errors
attempt_errors
successful_tool_calls（根部重复字段）
distinct_successful_tools（根部重复字段）
完整 provider tool result
完整 artifact 本机路径
```

工具调用计数集中在 `metrics`；完整 Answer 轨迹集中在 `messages`；人工读取使用 `tool_calls`。

## 严格 JSON

Compact writer 使用 `allow_nan=false`。所有 `NaN`、`Infinity` 和 `-Infinity` 在写入前转换为 JSON `null`，保证标准 JSONL 解析器可以逐行读取。
