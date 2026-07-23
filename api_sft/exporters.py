from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .common import iter_jsonl, write_json, write_jsonl
from .trajectory_exporters import _trl_content


SYSTEM_PROMPT=(Path(__file__).with_name("prompts")/"tool_execution_system_legacy.txt").read_text(encoding="utf-8").strip()


def export_datasets(verified_path: Path, output_dir: Path, rejected_path: Path|None=None, coverage_path: Path|None=None) -> dict[str,Any]:
    rows=list(iter_jsonl(verified_path)); output_dir.mkdir(parents=True,exist_ok=True); full=[]; trl=[]; total_tokens={"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}; winner_counts=Counter(); task_counts=Counter(); modality_counts=Counter()
    for row in rows:
        q=row["question_record"]; answer=row["selection"]["final_answer"]; assistant=json.dumps(answer,ensure_ascii=False)
        full.append(row)
        task_counts[q["task"]]+=1; modality_counts[q["input_mode"]]+=1
        winner_counts.update(row.get("selection",{}).get("selected_candidate_names",[]))
        trl.append({"messages":[{"role":"system","content":q.get("system_prompt") or SYSTEM_PROMPT},{"role":"user","content":_trl_content(q["question"],q["images"])},{"role":"assistant","content":assistant}],"images":q["images"]})
        for candidate in row.get("candidates",[]):
            for key in total_tokens: total_tokens[key]+=int(candidate.get("usage",{}).get(key,0) or 0)
    write_jsonl(output_dir/"final_full.jsonl",full); write_jsonl(output_dir/"train_trl_messages.jsonl",trl)
    rejected_count=sum(1 for _ in iter_jsonl(rejected_path)) if rejected_path and rejected_path.exists() else 0
    summary={"accepted_records":len(rows),"rejected_records":rejected_count,"acceptance_rate":round(len(rows)/max(1,len(rows)+rejected_count),4),"task_counts":dict(task_counts),"input_mode_counts":dict(modality_counts),"selected_candidate_counts":dict(winner_counts),"files":{"neutral_audit":"final_full.jsonl","trl":"train_trl_messages.jsonl"},"candidate_token_usage":total_tokens}
    if coverage_path and coverage_path.exists(): summary["coverage_report"]=json.loads(coverage_path.read_text(encoding="utf-8"))
    write_json(output_dir/"run_summary.json",summary); return summary
