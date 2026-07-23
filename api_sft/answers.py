from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .api_client import OpenAICompatibleClient, parse_json_object, user_message
from .common import append_jsonl, done_ids, iter_jsonl


ANSWER_SCHEMA={
 "analysis_mode":"text_only|text_then_image|image_text",
 "evidence_from_text":["由可见文本直接支持的证据"],
 "evidence_from_images":["由图像直接支持的证据；无图时必须为空"],
 "tool_plan":[{"step":1,"tool":"真实工具名","input_dependency":"所需前置结果","reason":"使用原因","uses_image":False,"expected_evidence":"要确认什么","branch_condition":"何时继续、换工具或停止"}],
 "final_answer":"完整中文专家答复",
 "model_recommendations":[{"model":"候选目录中的模型或明确标记的 baseline","rank":1,"reason":"适用原因","preprocessing":"前置处理","not_choose_when":"不选条件"}],
 "validation_plan":{"split":"rolling/expanding backtest","primary_metric":"主指标","secondary_metrics":["辅指标"],"leakage_checks":["泄漏检查"]},
 "risks_and_uncertainties":["不能过度断言的内容"],
}


def answer_messages(row: dict[str,Any]) -> list[dict[str,Any]]:
    catalog={"tools":[{"name":t["name"],"description":t.get("description"),"schema_name":t.get("schema_name"),"argument_keys":t.get("argument_keys",[])} for t in row["candidate_tools"]],
             "models":[{"name":m["name"],"package":m.get("package"),"tasks":m.get("tasks",[])} for m in row["candidate_models"]]}
    prompt={"question":row["question"],"input_mode":row["input_mode"],"modality_supervision":{"recommended_mode":row["recommended_mode"],"image_value":row["image_value"],"visual_reason":row["visual_reason"],"text_can_answer":row["text_can_answer"],"image_should_answer":row["image_should_answer"],"requires_statistical_confirmation":row["requires_statistical_confirmation"]},
            "available_catalog":catalog,"instructions":["只使用目录中存在的工具名","不能声称已执行尚未执行的工具","区分直接观察、统计确认和待验证假设","图像题必须利用图像；无图题不得声称看过图","工具计划必须有前后依赖和分支条件","不要臆造不可见精确数值"],"required_output_schema":ANSWER_SCHEMA}
    if row.get("retry_feedback"): prompt["retry_feedback_without_hidden_truth"]=row["retry_feedback"]
    return [{"role":"system","content":"你是资深时序分析、建模和多模态工具路由专家。只输出合法 JSON，不要 Markdown 代码块。"},user_message(json.dumps(prompt,ensure_ascii=False),row["images"])]


def _call_answer(row: dict[str,Any], cfg: dict[str,Any], timeout: int, retries: int) -> dict[str,Any]:
    name=cfg.get("name",cfg["model"]); client=OpenAICompatibleClient(cfg,timeout,retries)
    try:
        raw,usage,latency=client.complete(answer_messages(row)); parsed=parse_json_object(raw)
        return {"provider_name":name,"model":cfg["model"],"status":"ok","answer":parsed,"raw_content":raw,"usage":usage,"latency_seconds":round(latency,3)}
    except Exception as exc:
        return {"provider_name":name,"model":cfg.get("model"),"status":"error","error":str(exc),"usage":{},"latency_seconds":None}


def generate_answers(input_path: Path, output_path: Path, vlm_configs: list[dict[str,Any]], concurrency: int=4, timeout: int=120, retries: int=3, resume: bool=False, limit: int|None=None) -> None:
    rows=list(iter_jsonl(input_path)); rows=rows[:limit] if limit else rows
    legacy=[row["id"] for row in rows if row.get("question_spec_version") not in {"3.0","3.1","4.0"}]
    if legacy:
        preview=", ".join(legacy[:3])
        raise RuntimeError(f"Legacy Question records detected ({preview}). Regenerate QuestionSpec and questions with the V4 pipeline before continuing.")
    blocked=[row["id"] for row in rows if row.get("trajectory_requirement")=="tool_execution"]
    if blocked:
        preview=", ".join(blocked[:3])
        raise RuntimeError(f"Question records require real tool-execution trajectories ({preview}). The legacy answer generator only writes textual tool plans and is intentionally blocked until trajectory generation is implemented.")
    if output_path.exists() and not resume: output_path.unlink()
    completed=done_ids(output_path) if resume else set()
    for row in rows:
        if row["id"] in completed: continue
        candidates=[]
        with ThreadPoolExecutor(max_workers=min(concurrency,max(1,len(vlm_configs)))) as pool:
            futures=[pool.submit(_call_answer,row,cfg,timeout,retries) for cfg in vlm_configs]
            for future in as_completed(futures): candidates.append(future.result())
        append_jsonl(output_path,{"id":row["id"],"question_record":row,"candidate_count":len(candidates),"successful_candidate_count":sum(c["status"]=="ok" for c in candidates),"candidates":candidates})


def selector_messages(record: dict[str,Any]) -> list[dict[str,Any]]:
    q=record["question_record"]; candidates=[{"provider_name":c["provider_name"],"model":c["model"],"answer":c.get("answer")} for c in record["candidates"] if c["status"]=="ok"]
    payload={"task":"比较多个 VLM 候选，选择或融合为一个最终 SFT 答案。你看不到隐藏真值，不得补充候选和可见材料都不支持的事实。","question":q["question"],"input_mode":q["input_mode"],"visible_context":q["visible_context"],"modality_supervision":{k:q[k] for k in ["recommended_mode","image_value","visual_reason","text_can_answer","image_should_answer","requires_statistical_confirmation"]},
             "allowed_tools":[t["name"] for t in q["candidate_tools"]],"allowed_models":[m["name"] for m in q["candidate_models"]],"rubric":q["rubric"],"candidates":candidates,
             "required_output_schema":{"selected_candidate_names":["provider_name"],"candidate_scores":[{"provider_name":"name","correctness":0,"tool_routing":0,"image_grounding":0,"model_selection":0,"sft_value":0}],"selection_reason":"理由","final_answer":ANSWER_SCHEMA}}
    return [{"role":"system","content":"你是严格的时序 VLM SFT 答案裁决专家。每项分数为 0-5。只输出合法 JSON。"},user_message(json.dumps(payload,ensure_ascii=False),q["images"])]


def select_answers(input_path: Path, output_path: Path, selector_config: dict[str,Any], timeout: int=120, retries: int=3, resume: bool=False, limit: int|None=None) -> None:
    if output_path.exists() and not resume: output_path.unlink()
    completed=done_ids(output_path) if resume else set(); client=OpenAICompatibleClient(selector_config,timeout,retries); rows=list(iter_jsonl(input_path)); rows=rows[:limit] if limit else rows
    for record in rows:
        if record["id"] in completed: continue
        good=[c for c in record["candidates"] if c["status"]=="ok"]
        if not good:
            append_jsonl(output_path,{"id":record["id"],"status":"error","error":"no_successful_candidates","question_record":record["question_record"],"candidates":record["candidates"]}); continue
        try:
            raw,usage,latency=client.complete(selector_messages(record)); selected=parse_json_object(raw)
            append_jsonl(output_path,{"id":record["id"],"status":"ok","question_record":record["question_record"],"candidates":record["candidates"],"selection":selected,"selector_model":selector_config["model"],"selector_usage":usage,"selector_latency_seconds":round(latency,3)})
        except Exception as exc:
            append_jsonl(output_path,{"id":record["id"],"status":"error","error":str(exc),"question_record":record["question_record"],"candidates":record["candidates"]})
