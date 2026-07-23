from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .common import iter_jsonl, write_json, write_jsonl


def _without_image_placeholders(text: str) -> str:
    """Remove transport-specific placeholders replaced by TRL image blocks."""

    return re.sub(r"<image>\s*", "", text).strip()


def _trl_content(text: str, image_paths: list[str]) -> str | list[dict[str, str]]:
    if not image_paths:
        return text
    return [
        *({"type": "image"} for _ in image_paths),
        {"type": "text", "text": _without_image_placeholders(text)},
    ]


def _trl_tool_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert OpenAI wire arguments into Transformers/TRL Python dictionaries."""

    converted: list[dict[str, Any]] = []
    for call in calls:
        function = call["function"]
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("TRL tool-call arguments must decode to a JSON object")
        converted.append(
            {
                "type": "function",
                "function": {"name": function["name"], "arguments": arguments},
            }
        )
    return converted


def _trl_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project one neutral audit record into TRL conversational SFT format."""

    messages: list[dict[str, Any]] = []
    images: list[str] = []
    for message in record["messages"]:
        role = message["role"]
        message_images = [str(path) for path in message.get("images", [])]
        if role in {"system", "user"}:
            messages.append(
                {
                    "role": role,
                    "content": _trl_content(str(message.get("content", "")), message_images),
                }
            )
            images.extend(message_images)
        elif role == "assistant" and message.get("tool_calls"):
            item: dict[str, Any] = {
                "role": "assistant",
                "tool_calls": _trl_tool_calls(message["tool_calls"]),
            }
            decision = str(message.get("content") or "").strip()
            if decision:
                item["content"] = decision
            messages.append(item)
        elif role == "tool":
            messages.append(
                {
                    "role": "tool",
                    "name": str(message["name"]),
                    "content": str(message.get("content", "")),
                }
            )
            if message_images:
                messages.append(
                    {
                        "role": "user",
                        "content": _trl_content(
                            f"以下图像是工具 {message['name']} 刚生成的真实视觉结果，请将其视为上一条工具结果的一部分。",
                            message_images,
                        ),
                    }
                )
                images.extend(message_images)
        elif role == "assistant":
            messages.append({"role": "assistant", "content": str(message.get("content", ""))})
        else:
            raise ValueError(f"Unsupported neutral message role for TRL export: {role}")
    return {"messages": messages, "tools": record["tools"], "images": images}


def export_trajectory_datasets(verified_path: Path, output_dir: Path) -> dict[str, Any]:
    records = list(iter_jsonl(verified_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    trl = [_trl_record(record) for record in records]
    write_jsonl(output_dir / "trajectories.full.jsonl", records)
    write_jsonl(output_dir / "train_trl_tool_messages.jsonl", trl)
    task_counts = Counter(record["question_record"].get("task") for record in records)
    modality_counts = Counter(record["question_record"].get("input_mode") for record in records)
    tool_counts = Counter(event.get("name") for record in records for event in record.get("tool_events", []) if event.get("ok"))
    summary = {
        "accepted_records": len(records),
        "format": "trl_conversational_tool_sft_v1",
        "task_counts": dict(task_counts),
        "input_mode_counts": dict(modality_counts),
        "successful_tool_counts": dict(tool_counts),
        "files": {
            "neutral_audit": "trajectories.full.jsonl",
            "trl": "train_trl_tool_messages.jsonl",
        },
    }
    write_json(output_dir / "trajectory_run_summary.json", summary)
    return summary
