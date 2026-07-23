from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .api_client import OpenAICompatibleClient, parse_json_object, user_message
from .common import append_jsonl, done_ids, iter_jsonl, read_json, write_jsonl


REQUIRED_ANSWER_KEYS={"analysis_mode","evidence_from_text","evidence_from_images","tool_plan","final_answer","model_recommendations","validation_plan","risks_and_uncertainties"}
BASELINES={"SeasonalNaive","Naive","Drift","MovingAverage","ARIMA","ETS","Croston","TSB","baseline"}


def deterministic_review(record: dict[str,Any], valid_tools: set[str], valid_models: set[str]) -> dict[str,Any]:
    if record.get("status")!="ok": return {"score":0.0,"passed":False,"flags":["selection_error"]}
    q=record["question_record"]; selection=record.get("selection",{}); answer=selection.get("final_answer",{}); flags=[]
    if not isinstance(answer,dict): return {"score":0.0,"passed":False,"flags":["final_answer_not_object"]}
    schema_score=len(REQUIRED_ANSWER_KEYS&set(answer))/len(REQUIRED_ANSWER_KEYS)
    plans=answer.get("tool_plan",[]); used=[step.get("tool") for step in plans if isinstance(step,dict)]; invalid_tools=sorted({name for name in used if name and name not in valid_tools})
    if invalid_tools: flags.append("invalid_tools:"+",".join(invalid_tools))
    ordered=all(isinstance(step,dict) and step.get("step")==i+1 and step.get("reason") and step.get("branch_condition") for i,step in enumerate(plans)) if plans else False
    if not ordered: flags.append("tool_plan_not_ordered_or_incomplete")
    primary=set(q.get("primary_tools",[])); relevant=not primary or bool(primary&set(used))
    if primary and not relevant: flags.append("tool_plan_missing_subtask_relevant_tool")
    image_evidence=answer.get("evidence_from_images",[])
    if q["input_mode"]=="text_only" and image_evidence: flags.append("text_only_claims_image_evidence")
    if q["input_mode"]=="image_text" and not image_evidence: flags.append("missing_image_grounding")
    recommendations=answer.get("model_recommendations",[]); invalid_models=[]
    for item in recommendations if isinstance(recommendations,list) else []:
        if isinstance(item,dict):
            name=str(item.get("model",""));
            if name and name not in valid_models and name not in BASELINES and "baseline" not in name.lower(): invalid_models.append(name)
    if invalid_models: flags.append("invalid_models:"+",".join(sorted(set(invalid_models))))
    answer_len=len(str(answer.get("final_answer",""))); length_score=1.0 if 300<=answer_len<=4000 else .5 if answer_len>=150 else .1
    evidence_score=1.0 if answer.get("evidence_from_text") and answer.get("risks_and_uncertainties") else .4
    modality_score=0.0 if any(f in flags for f in ["text_only_claims_image_evidence","missing_image_grounding"]) else 1.0
    tool_score=1.0 if ordered and not invalid_tools and relevant else .6 if ordered and not invalid_tools else .2
    score=.25*schema_score+.25*tool_score+.2*modality_score+.15*evidence_score+.15*length_score
    fatal=any(f.startswith("invalid_tools") or f.startswith("invalid_models") for f in flags) or schema_score<1 or bool(primary and not relevant)
    return {"score":round(score,4),"passed":score>=.75 and not fatal,"flags":flags,"subscores":{"schema":round(schema_score,4),"tool_plan":tool_score,"modality":modality_score,"evidence":evidence_score,"length":length_score}}


def verifier_messages(record: dict[str,Any], truth: dict[str,Any], deterministic: dict[str,Any]) -> list[dict[str,str]]:
    q=record["question_record"]
    payload={"instruction":"只验证最终答案，不得重写、补充或返回修订答案。检查其是否符合隐藏生成真值、可见证据边界和时序常识。只输出评分与错误。","question":q["question"],"input_mode":q["input_mode"],"visible_context":q["visible_context"],"hidden_truth_for_verification_only":truth,"final_answer":record["selection"].get("final_answer"),"deterministic_review":deterministic,
             "required_output_schema":{"correctness":0,"evidence_boundary":0,"tool_routing":0,"image_grounding":0,"overall_score":0,"passed":False,"factual_errors":[],"evidence_violations":[],"notes":"质检说明；不得包含修订答案"}}
    return [{"role":"system","content":"你是只读质量验证器。分数范围 0-5。禁止改写答案。只输出合法 JSON。"},user_message(json.dumps(payload,ensure_ascii=False),q["images"])]


def verify_records(selected_path: Path, scenarios_path: Path, tools_path: Path, models_path: Path, verified_path: Path, rejected_path: Path, minimum_score: float=.75, verifier_config: dict[str,Any]|None=None, timeout: int=120, retries: int=3, resume: bool=False, limit: int|None=None) -> None:
    if not resume:
        if verified_path.exists(): verified_path.unlink()
        if rejected_path.exists(): rejected_path.unlink()
    completed=done_ids(verified_path)|done_ids(rejected_path) if resume else set(); scenarios={r["id"]:r for r in iter_jsonl(scenarios_path)}; valid_tools={t["name"] for t in read_json(tools_path)["tools"]}; valid_models={m["name"] for m in read_json(models_path)["models"]}
    client=OpenAICompatibleClient(verifier_config,timeout,retries) if verifier_config else None; records=list(iter_jsonl(selected_path)); records=records[:limit] if limit else records
    for record in records:
        if record["id"] in completed: continue
        det=deterministic_review(record,valid_tools,valid_models); judge=None; score=det["score"]
        if client and record.get("status")=="ok":
            truth=read_json(Path(scenarios[record["question_record"]["scenario_id"]]["truth_path"]))
            try:
                raw,usage,latency=client.complete(verifier_messages(record,truth,det)); judge=parse_json_object(raw); judge["usage"]=usage; judge["latency_seconds"]=round(latency,3); score=.45*det["score"]+.55*(float(judge.get("overall_score",0))/5)
            except Exception as exc: judge={"passed":False,"overall_score":0,"error":str(exc)}; score=0
        passed=det["passed"] and score>=minimum_score and (judge is None or bool(judge.get("passed")))
        out={"id":record["id"],"status":"passed" if passed else "rejected","final_score":round(score,4),"deterministic_review":det,"truth_verifier_review":judge,"question_record":record.get("question_record"),"selection":record.get("selection"),"candidates":record.get("candidates",[])}
        append_jsonl(verified_path if passed else rejected_path,out)
