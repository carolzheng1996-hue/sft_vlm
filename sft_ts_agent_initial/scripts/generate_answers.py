#!/usr/bin/env python3
"""Generate expert SFT answers with an OpenAI-compatible Chat Completions API."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body["choices"][0]["message"]["content"]


def selection_messages(seed: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, str]]:
    compact_candidates = [
        {
            "candidate_id": c["candidate_id"],
            "parsed": c["assistant_obj"],
            "raw_excerpt": c["raw_content"][:1200],
        }
        for c in candidates
    ]
    prompt = {
        "task": "从多个候选 SFT 答案中选择或融合一个最终答案。",
        "selection_criteria": [
            "必须遵循用户问题和 required_output_schema",
            "工具调用顺序合理，不能跳过数据读取/画像/诊断的必要步骤",
            "image_text 样本必须明确哪些判断来自图像，哪些来自统计/元数据",
            "模型选择要结合数据模式、任务目标、验证方式和成本，不要只罗列模型名",
            "不能臆造可见文件中没有的具体数值",
            "答案应有适合 SFT 的中等难度和可迁移专家策略",
        ],
        "seed_metadata": {
            "id": seed.get("id"),
            "category": seed.get("category"),
            "input_mode": seed.get("input_mode"),
            "expert_trace": seed.get("expert_trace"),
        },
        "question": seed.get("question"),
        "candidates": compact_candidates,
        "required_output_schema": {
            "selected_candidate_ids": ["候选 id，可多选表示融合"],
            "selection_reason": "为什么这个版本最好",
            "tool_plan": [],
            "answer": "最终中文专家答案",
            "model_or_strategy_recommendation": [],
            "risks_and_uncertainties": [],
            "quality_checks": [],
        },
    }
    return [
        {
            "role": "system",
            "content": "你是 SFT 数据质检与答案融合专家。只输出合法 JSON，不要 Markdown 代码块。",
        },
        {
            "role": "user",
            "content": json.dumps(prompt, ensure_ascii=False, indent=2),
        },
    ]


def normalize_assistant_content(raw_content: str) -> dict[str, Any]:
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        return {
            "tool_plan": [],
            "answer": raw_content,
            "model_or_strategy_recommendation": [],
            "risks_and_uncertainties": ["模型未返回合法 JSON，需人工清洗。"],
            "quality_checks": [],
        }


def build_sft_record(
    seed: dict[str, Any],
    assistant_obj: dict[str, Any],
    raw_content: str,
    candidates: list[dict[str, Any]] | None = None,
    selection_raw_content: str | None = None,
) -> dict[str, Any]:
    assistant_text = json.dumps(assistant_obj, ensure_ascii=False, indent=2)
    return {
        "id": seed["id"],
        "messages": seed["messages"] + [{"role": "assistant", "content": assistant_text}],
        "images": seed.get("images", []),
        "metadata": {
            **seed.get("metadata", {}),
            "case_id": seed.get("case_id"),
            "category": seed.get("category"),
            "input_mode": seed.get("input_mode"),
            "answer_generated": True,
        },
        "expert_trace": seed.get("expert_trace", {}),
        "raw_assistant_content": raw_content,
        "generation_candidates": candidates or [],
        "selection_raw_content": selection_raw_content,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sample-temperature", type=float, default=None)
    parser.add_argument("--select-best", action="store_true")
    parser.add_argument("--selection-temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    done_ids = set()
    if args.resume and args.output.exists():
        for row in iter_jsonl(args.output):
            done_ids.add(row["id"])

    seeds = list(iter_jsonl(args.input))
    if args.limit is not None:
        seeds = seeds[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as out:
        for idx, seed in enumerate(seeds, start=1):
            if seed["id"] in done_ids:
                continue
            try:
                candidates = []
                sample_temperature = args.temperature if args.sample_temperature is None else args.sample_temperature
                for sample_idx in range(args.num_samples):
                    raw = post_chat_completion(
                        args.base_url,
                        api_key,
                        args.model,
                        seed["messages"],
                        sample_temperature,
                        args.timeout,
                    )
                    candidates.append(
                        {
                            "candidate_id": f"sample_{sample_idx + 1}",
                            "raw_content": raw,
                            "assistant_obj": normalize_assistant_content(raw),
                        }
                    )
                    if args.sleep > 0 and sample_idx + 1 < args.num_samples:
                        time.sleep(args.sleep)

                selection_raw = None
                if args.select_best and len(candidates) > 1:
                    selection_raw = post_chat_completion(
                        args.base_url,
                        api_key,
                        args.model,
                        selection_messages(seed, candidates),
                        args.selection_temperature,
                        args.timeout,
                    )
                    assistant_obj = normalize_assistant_content(selection_raw)
                    raw = selection_raw
                else:
                    assistant_obj = candidates[0]["assistant_obj"]
                    raw = candidates[0]["raw_content"]
                record = build_sft_record(seed, assistant_obj, raw, candidates, selection_raw)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[{idx}/{len(seeds)}] ok {seed['id']} samples={len(candidates)}")
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="replace")
                print(f"[{idx}/{len(seeds)}] http_error {seed['id']}: {e.code} {body}")
            except Exception as e:
                print(f"[{idx}/{len(seeds)}] error {seed['id']}: {e}")
            if args.sleep > 0:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
