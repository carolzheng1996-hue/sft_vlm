from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .common import iter_jsonl
from .compact_json import write_json, write_jsonl
from .trajectory_exporters import _trl_record


def export_trajectory_datasets(verified_path: Path, output_dir: Path) -> dict[str, Any]:
    records = list(iter_jsonl(verified_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    trl = [_trl_record(record) for record in records]
    write_jsonl(output_dir / "trajectories.compact.jsonl", records)
    write_jsonl(output_dir / "train_trl_tool_messages.jsonl", trl)
    task_counts = Counter((record.get("question") or {}).get("task") for record in records)
    modality_counts = Counter((record.get("question") or {}).get("input_mode") for record in records)
    tool_counts = Counter(
        event.get("name")
        for record in records
        for event in record.get("tool_calls", [])
        if event.get("ok")
    )
    summary = {
        "accepted_records": len(records),
        "format": "trl_conversational_tool_sft_v1",
        "task_counts": {key: value for key, value in task_counts.items() if key is not None},
        "input_mode_counts": {key: value for key, value in modality_counts.items() if key is not None},
        "successful_tool_counts": {key: value for key, value in tool_counts.items() if key is not None},
        "files": {
            "inspection": "trajectories.compact.jsonl",
            "trl": "train_trl_tool_messages.jsonl",
        },
    }
    write_json(output_dir / "trajectory_run_summary.json", summary)
    return summary
