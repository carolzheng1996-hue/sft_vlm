# 单候选 SFT 多轮轨迹生成器

该包与原有 `api_sft` 双候选流程隔离。每道题只生成一个真实工具调用轨迹，
不读取 selector 配置，也不执行候选竞争。多轮工具循环继续复用
`api_sft.trajectories.generate_one_trajectory`，硬校验继续原样调用
`api_sft.trajectory_verify.deterministic_trajectory_review`。

单候选包按 OpenAI-compatible `/chat/completions` 协议调用 API 平台，不根据
GPT、Claude、Kimi、GLM 或 Qwen 等模型名称切换协议。模型只生成业务参数，
runtime 自动注入 `session_id`；请求统一使用可移植的临时 Schema，JSON 字符串
容器只有通过原始 Schema 校验后才会恢复。正式 adaptive 轨迹每轮仍只允许一个
工具调用，多调用响应只纠错一次且不会被执行。

## 配置

```bash
cp api_sft_single_candidate/config.example.yaml api_sft_single_candidate/config.yaml
```

在 `config.yaml` 中配置唯一的 `models.candidate`，必填字段只有 `base_url`、
`model` 和直接填写的 `api_key`。`config.yaml` 已被本目录的 `.gitignore` 忽略；
API Key 不会写入日志、audit 或训练数据。

```yaml
models:
  candidate:
    base_url: https://your-api-platform.example/v1
    model: kimi-k3
    api_key: replace-with-your-api-key
    # 可选；未配置时不会向模型发送 temperature
    temperature: 0.2
```

切换模型只需修改 `model`。平台提供的 Claude 模型只要使用同一个
`/chat/completions` 接口，也按上述配置调用；本包不实现 Anthropic 原生
`/v1/messages`。`base_url` 可以填写 `/v1` 根地址，也可以填写完整的
`/chat/completions` 地址，客户端不会重复拼接路径。

`trajectory_generation.max_concurrency` 控制不同 Question 之间的并发数，默认值为
`4`。它不会改变单道题内部的多轮工具调用顺序；每道题仍然每轮最多执行一个工具
调用。当前配置默认读取并发 Question 生成器的产物：
`api_sft/output/concurrent_questions/questions.final.jsonl`。

导出的 `messages`、`tools` 和 tool-call 参数不含运行期 `session_id`。原始
`claude_tsa` 工具 Schema 与执行校验保持不变，会话值仅在实际执行前注入。

## 运行

生成、校验并导出：

```bash
python -m api_sft_single_candidate \
  --config api_sft_single_candidate/config.yaml \
  run-trajectories --fresh --max-concurrency 4
```

小规模试跑可以限制题数：

```bash
python -m api_sft_single_candidate \
  --config api_sft_single_candidate/config.yaml \
  run-trajectories --fresh --limit 8 --max-concurrency 4
```

命令行的 `--max-concurrency` 优先于 YAML 配置，必须大于等于 1。建议从 4 开始；
如果模型服务出现限流、超时或本机工具执行资源紧张，可降到 2 或 1。并发只发生在
不同 Question 之间，同一 Question 的 API 重试、工具调用和最终回答仍保持顺序。
程序会在每道题完成后打印进度 JSON，并立即写入 raw/audit checkpoint。例如：

```text
{"status":"progress","completed":37,"total":622,"percent":5.9,"accepted":36,"rejected":1,...}
```

正式并发前会先在本地校验相关工具 Schema，并将第一条真实 Question 作为
canary 串行执行；canary 结果会直接保存，不会重复请求。普通质量 reject 不会
阻塞后续题目，但鉴权、URL、Schema、协议或持续网络错误会停止继续提交任务，
避免把平台故障写成整批 rejected。修复配置或服务后使用 `--resume` 继续。

中断后继续：

```bash
python -m api_sft_single_candidate \
  --config api_sft_single_candidate/config.yaml \
  run-trajectories --resume --max-concurrency 4
```

`--resume` 会按 Question ID 对齐 raw 和 audit，只复用两边都完整存在的记录；如果
上次中断只写入了其中一个文件，孤立记录会被清理并重新生成。`--fresh` 只归档
`api_sft/output_single_candidate/` 下的单候选产物，不会移动 Question 生成器或双候选
目录中的文件。如果配置中的 `model` 与已有记录不同，`--resume` 会拒绝混合模型，
此时应使用 `--fresh`。

也可以分别执行 `generate-trajectories`、`verify-trajectories` 和
`export-trajectories`。

默认输出位于 `api_sft/output_single_candidate/`：

- `trajectories.raw.jsonl`
- `trajectories.generation_audit.jsonl`
- `trajectories.verified.jsonl`
- `trajectories.rejected.jsonl`
- `trajectory_exports/trajectories.compact.jsonl`
- `trajectory_exports/train_trl_tool_messages.jsonl`

`--fresh` 只归档上述单候选输出；不会移动或覆盖原有 `api_sft/output/`
中的双候选轨迹。
