from __future__ import annotations

import json
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_sft.common import write_jsonl
from api_sft.cli import _assert_resumable_question_formats, _assert_resumable_trajectory_formats, archive_question_run, archive_trajectory_run
from api_sft.trajectories import COMPETITION_AUDIT_FORMAT_VERSION, TRAJECTORY_FORMAT_VERSION, TrajectoryGenerationError, _generate_candidate, _search_model_catalog, _select_with_judge, generate_one_trajectory, generate_trajectories
from api_sft.trajectory_exporters import _trl_record, export_trajectory_datasets
from api_sft.trajectory_verify import deterministic_trajectory_review, verify_trajectories


class FakeClient:
    def __init__(self) -> None:
        self.config = {"model": "mock-tool-model"}
        self.turn = 0

    def complete_message(self, messages, tools=None, tool_choice=None):
        self.turn += 1
        if self.turn == 1:
            message = {
                "role": "assistant",
                "content": "先核验原始表结构，确认当前摘要是否可靠。",
                "tool_calls": [{"id": "call_profile", "type": "function", "function": {"name": "data_profile", "arguments": json.dumps({"session_id": "session_test", "uri": "uploads/dataset.csv"})}}],
            }
        else:
            message = {
                "role": "assistant",
                "content": "真实工具结果表明数据字段与目标序列均可读取，当前样本没有发现阻断后续分析的结构问题。建议先使用低成本、可解释的基线完成时间顺序回测，再根据残差是否仍有稳定趋势或周期决定是否增加复杂度。业务上低估代价更高，因此模型选择后应在滚动验证集上评估低估率与分位数损失，并通过分位数预测或仅由历史验证确定的校正量降低低估风险；不能从当前画像直接声称未来精度已经达标。",
            }
        return message, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}, 0.01, "tool_calls" if self.turn < 2 else "stop"


class FakeRuntime:
    def __init__(self) -> None:
        self.session = "session_test"

    def start(self, question_id, data_path, allowed_names):
        return {
            "session_id": self.session,
            "workspace_path": "/tmp/session_test",
            "dataset_uri": "uploads/dataset.csv",
            "source_dataset_sha256": "a",
            "staged_dataset_sha256": "a",
            "initial_artifact_count": 0,
        }

    def tools_for_model(self):
        def tool(name, properties, required):
            properties = {"session_id": {"type": "string", "const": self.session}, **properties}
            return {"type": "function", "function": {"name": name, "description": name, "parameters": {"type": "object", "properties": properties, "required": ["session_id", *required]}}}

        return [
            tool("data_profile", {"uri": {"type": "string"}}, ["uri"]),
            tool("summary_stats", {"channel": {"type": "string"}}, ["channel"]),
        ]

    def execute(self, name, arguments):
        summary = {"rows": 100, "columns": ["time", "series_id", "value"]} if name == "data_profile" else {"n": 100, "mean": 12.0, "std": 2.0, "channel": "s01"}
        structured = {"ok": True, "summary": summary}
        if name == "data_profile":
            structured["artifact"] = {"uri": "outputs/reports/profile.json", "path": "/private/tmp/session_test/outputs/reports/profile.json"}
        return {
            "ok": True,
            "result": {"content": [{"type": "text", "text": "done"}], "structuredContent": structured},
            "created_artifacts": [],
            "image_paths": [],
        }

    def source_snapshot(self):
        return {"repo_root": "/repo", "git_head": "abc", "git_dirty": False}


def question_row() -> dict:
    return {
        "id": "q1",
        "format_version": "question_runtime_v1",
        "spec_ref": {"version": "4.0", "hash": "question-hash"},
        "task": {
            "category": "data_profile",
            "subtask_id": "profile_schema_frequency_index",
            "goal": "forecast",
            "input_mode": "text_only",
            "model_catalog_scope": "none",
        },
        "prompt": {
            "system_prompt_id": "tsa_tool_execution_v2",
            "user_request": "请分析数据结构并决定后续预测策略。",
        },
        "resources": {
            "dataset": {"path": "/absolute/source.csv", "format": "csv"},
            "images": [],
        },
        "allowed_tools": ["data_profile", "summary_stats"],
    }


class TrajectoryTests(unittest.TestCase):
    def test_fresh_archive_moves_question_outputs_recoverably(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mapping={
                "questions":root/"questions.final.jsonl",
                "question_audit":root/"questions.audit.jsonl",
                "question_rejected":root/"questions.rejected.jsonl",
                "coverage":root/"coverage_report.json",
                "question_archives":root/"archives",
            }
            write_jsonl(mapping["questions"],[{"id":"q1"}])
            write_jsonl(mapping["question_audit"],[{"id":"q1"}])
            mapping["coverage"].write_text("{}",encoding="utf-8")
            archive=archive_question_run(mapping)
            self.assertIsNotNone(archive)
            self.assertFalse(mapping["questions"].exists())
            manifest=json.loads((archive/"archive_manifest.json").read_text())
            self.assertEqual({entry["archived_as"] for entry in manifest["entries"]},{"questions.final.jsonl","questions.audit.jsonl","coverage_report.json"})
            self.assertTrue((archive/"questions.final.jsonl").exists())

    def test_resume_rejects_stale_or_incomplete_question_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); final=root/"questions.final.jsonl"; audit=root/"questions.audit.jsonl"
            write_jsonl(final,[{"id":"q1","format_version":"question_contract_v2"}])
            with self.assertRaisesRegex(RuntimeError,"use --fresh"):
                _assert_resumable_question_formats({"questions":final,"question_audit":audit})

    def test_fresh_archive_moves_all_trajectory_outputs_recoverably(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            mapping={
                "trajectories":root/"raw.jsonl",
                "trajectory_audit":root/"audit.jsonl",
                "trajectories_verified":root/"verified.jsonl",
                "trajectories_rejected":root/"rejected.jsonl",
                "trajectory_workspaces":root/"workspaces",
                "trajectory_exports":root/"exports",
                "trajectory_archives":root/"archives",
            }
            write_jsonl(mapping["trajectories"],[{"id":"q1"}]); write_jsonl(mapping["trajectory_audit"],[{"id":"q1"}])
            mapping["trajectory_workspaces"].mkdir(); (mapping["trajectory_workspaces"]/"session.txt").write_text("x")
            mapping["trajectory_exports"].mkdir(); (mapping["trajectory_exports"]/"train.jsonl").write_text("{}\n")
            archive=archive_trajectory_run(mapping)
            self.assertIsNotNone(archive)
            self.assertFalse(mapping["trajectories"].exists()); self.assertFalse(mapping["trajectory_workspaces"].exists())
            manifest=json.loads((archive/"archive_manifest.json").read_text())
            self.assertEqual({entry["archived_as"] for entry in manifest["entries"]},{"raw.jsonl","audit.jsonl","workspaces","exports"})
            self.assertTrue((archive/"workspaces"/"session.txt").exists())

    def test_resume_rejects_stale_trajectory_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); raw=root/"raw.jsonl"; audit=root/"audit.jsonl"
            write_jsonl(raw,[{"id":"q1","format_version":"tool_trajectory_v2"}])
            write_jsonl(audit,[{"id":"q1","format_version":"trajectory_competition_audit_v1"}])
            with self.assertRaisesRegex(RuntimeError,"use --fresh"):
                _assert_resumable_trajectory_formats({"trajectories":raw,"trajectory_audit":audit})

    def test_mock_tool_loop_verifies_and_exports_neutral_and_trl(self):
        record = generate_one_trajectory(question_row(), FakeClient(), FakeRuntime())
        review = deterministic_trajectory_review(record)
        self.assertTrue(review["passed"], review)
        self.assertEqual(record["distinct_successful_tools"], ["data_profile"])
        system=record["messages"][0]["content"]
        for forbidden in ["prepared_channels","primary_tools","model_route","matched_subclass"]:
            self.assertNotIn(forbidden,system)
        self.assertNotIn("/absolute/source.csv", json.dumps(record["messages"], ensure_ascii=False))
        self.assertNotIn("/private/tmp/session_test", json.dumps(record["messages"], ensure_ascii=False))
        self.assertEqual(record["messages"][1]["content"],"请分析数据结构并决定后续预测策略。\n\n可访问的数据资源：\n- dataset_path: uploads/dataset.csv")
        self.assertNotIn("已提供的结构化材料",record["messages"][1]["content"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw.jsonl"
            verified = root / "verified.jsonl"
            rejected = root / "rejected.jsonl"
            write_jsonl(raw, [record])
            verify_trajectories(raw, verified, rejected)
            self.assertTrue(verified.exists())
            summary = export_trajectory_datasets(verified, root / "export")
            self.assertEqual(summary["accepted_records"], 1)
            self.assertEqual(summary["format"], "trl_conversational_tool_sft_v1")
            self.assertTrue((root / "export" / "trajectories.full.jsonl").exists())
            trl = json.loads((root / "export" / "train_trl_tool_messages.jsonl").read_text())
            self.assertEqual(trl["messages"][2]["tool_calls"][0]["function"]["name"], "data_profile")
            self.assertIsInstance(trl["messages"][2]["tool_calls"][0]["function"]["arguments"], dict)
            self.assertEqual(trl["messages"][3]["name"], "data_profile")
            self.assertNotIn("tool_call_id", trl["messages"][3])
            self.assertEqual(trl["images"], [])

    def test_trl_export_uses_vision_blocks_and_top_level_images(self):
        record = generate_one_trajectory(question_row(), FakeClient(), FakeRuntime())
        record["messages"][1]["content"] = "<image>\n请结合初始图片分析。"
        record["messages"][1]["images"] = ["/tmp/initial.png"]
        record["messages"][3]["images"] = ["/tmp/tool_plot.png"]
        trl = _trl_record(record)
        self.assertEqual(trl["images"], ["/tmp/initial.png", "/tmp/tool_plot.png"])
        self.assertEqual(trl["messages"][1]["content"][0], {"type": "image"})
        self.assertNotIn("<image>", trl["messages"][1]["content"][1]["text"])
        self.assertEqual(trl["messages"][3]["role"], "tool")
        self.assertEqual(trl["messages"][4]["role"], "user")
        self.assertEqual(trl["messages"][4]["content"][0], {"type": "image"})

    def test_invalid_tool_arguments_are_returned_and_can_be_repaired(self):
        class RepairClient(FakeClient):
            def complete_message(self, messages, tools=None, tool_choice=None):
                self.turn += 1
                if self.turn == 1:
                    message={"role":"assistant","content":"先尝试画像。","tool_calls":[{"id":"bad","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"wrong","uri":"uploads/dataset.csv"})}}]}
                elif self.turn == 2:
                    message={"role":"assistant","content":"修正 session 后重试。","tool_calls":[{"id":"good","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"session_test","uri":"uploads/dataset.csv"})}}]}
                else:
                    message={"role":"assistant","content":"画像工具已在修正参数后成功返回，字段和行数证据可以支持当前结构判断；第一次失败不作为数据结论。后续若进入具体序列统计，再显式转换所需列。"}
                return message,{"total_tokens":5},.01,"tool_calls" if self.turn<3 else "stop"
        record=generate_one_trajectory(question_row(),RepairClient(),FakeRuntime())
        self.assertFalse(record["tool_events"][0]["ok"])
        self.assertEqual(record["tool_events"][0]["error_type"],"schema_validation")
        self.assertTrue(record["tool_events"][1]["ok"])
        review=deterministic_trajectory_review(record)
        self.assertTrue(review["passed"],review)
        self.assertTrue(any(item.startswith("recovered_invalid_tool_arguments") for item in review["warnings"]))

    def test_repeated_failure_signature_terminates_candidate(self):
        class Repeater(FakeClient):
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1
                return {"role":"assistant","content":"重试同一个错误参数。","tool_calls":[{"id":f"bad_{self.turn}","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"wrong","uri":"uploads/dataset.csv"})}}]},{"total_tokens":1},.01,"tool_calls"
        with self.assertRaisesRegex(TrajectoryGenerationError,"Repeated tool failure signature") as raised:
            generate_one_trajectory(question_row(),Repeater(),FakeRuntime())
        self.assertEqual(len(raised.exception.partial_record["tool_events"]),2)
        detail=raised.exception.partial_record["tool_events"][0]["result"]["structuredContent"]["error"]
        self.assertIn("required_fields",detail); self.assertIn("allowed_fields",detail)

    def test_changed_schema_failure_arguments_are_allowed_to_recover(self):
        class RepairingClient(FakeClient):
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1
                if self.turn<=2:
                    uri="uploads/first.csv" if self.turn==1 else "uploads/second.csv"
                    message={"role":"assistant","content":"修正参数。","tool_calls":[{"id":f"bad_{self.turn}","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"wrong","uri":uri})}}]}
                elif self.turn==3:
                    message={"role":"assistant","content":"使用正确会话。","tool_calls":[{"id":"good","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"session_test","uri":"uploads/dataset.csv"})}}]}
                else:
                    message={"role":"assistant","content":"参数修正后画像成功，现有字段证据足以完成结构判断；前两次失败不作为数据结论。"}
                return message,{"total_tokens":1},.01,"tool_calls" if self.turn<4 else "stop"
        record=generate_one_trajectory(question_row(),RepairingClient(),FakeRuntime())
        self.assertEqual([event["ok"] for event in record["tool_events"]],[False,False,True])

    def test_tool_budget_warns_at_seven_and_forces_final_after_eight(self):
        class BudgetClient(FakeClient):
            def __init__(self): super().__init__(); self.choices=[]
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1; self.choices.append(tool_choice)
                if self.turn<=8:
                    message={"role":"assistant","content":"继续收集必要证据。","tool_calls":[{"id":f"call_{self.turn}","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"session_test","uri":"uploads/dataset.csv"})}}]}
                else:
                    message={"role":"assistant","content":"八次真实工具结果已覆盖结构与质量证据，现在停止调用并给出有依据的最终判断。"}
                return message,{"total_tokens":1},.01,"tool_calls" if self.turn<=8 else "stop"
        client=BudgetClient(); record=generate_one_trajectory(question_row(),client,FakeRuntime())
        self.assertEqual(client.choices[-1],"none")
        self.assertEqual(record["tool_events"][6]["trajectory_budget"]["remaining_tool_calls"],1)
        self.assertEqual(record["tool_events"][7]["trajectory_budget"]["remaining_tool_calls"],0)
        self.assertTrue(deterministic_trajectory_review(record)["passed"])

    def test_provider_tool_call_after_exhausted_budget_is_audited(self):
        class IgnoringClient(FakeClient):
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1
                message={"role":"assistant","content":"继续调用。","tool_calls":[{"id":f"call_{self.turn}","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"session_test","uri":"uploads/dataset.csv"})}}]}
                return message,{"total_tokens":1},.01,"tool_calls"
        with self.assertRaises(TrajectoryGenerationError) as raised:
            generate_one_trajectory(question_row(),IgnoringClient(),FakeRuntime())
        self.assertEqual(raised.exception.error_type,"tool_call_after_budget_exhausted")
        self.assertEqual(len(raised.exception.partial_record["tool_events"]),8)

    def test_failed_candidate_keeps_full_partial_attempts(self):
        class Repeater(FakeClient):
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1
                message={"role":"assistant","content":"重复错误参数。","tool_calls":[{"id":f"bad_{self.turn}","type":"function","function":{"name":"data_profile","arguments":json.dumps({"session_id":"wrong","uri":"uploads/dataset.csv"})}}]}
                return message,{"total_tokens":1},.01,"tool_calls"
        with patch("api_sft.trajectories.OpenAICompatibleClient",return_value=Repeater()),patch("api_sft.trajectories.ClaudeTsaToolRuntime",return_value=FakeRuntime()):
            candidate=_generate_candidate(question_row(),{"model":"m"},0,Path("/repo"),Path("/tmp/ws"),[],2,1,1,8,16000)
        self.assertEqual(candidate["status"],"error")
        self.assertEqual(len(candidate["failed_attempts"]),2)
        self.assertTrue(all(len(attempt["tool_events"])==2 for attempt in candidate["failed_attempts"]))

    @unittest.skipUnless(sys.version_info >= (3, 11) and Path("/Users/monychen/Documents/tsa/claude_tsa/pulsar/tsa/tools").is_dir(), "live claude_tsa runtime requires Python 3.11+")
    def test_live_runtime_starts_empty_then_data_convert_creates_channels(self):
        from api_sft.tool_runtime import ClaudeTsaToolRuntime

        with tempfile.TemporaryDirectory() as tmp:
            data=Path(tmp)/"wide.csv"
            data.write_text("time,s01,s02\n0,1,3\n1,2,4\n",encoding="utf-8")
            runtime = ClaudeTsaToolRuntime(Path("/Users/monychen/Documents/tsa/claude_tsa"), Path(tmp))
            setup = runtime.start("live_test", data, {"list_channels","data_convert"})
            initial = runtime.execute("list_channels", {"session_id": setup["session_id"]})
            converted = runtime.execute("data_convert", {"session_id":setup["session_id"],"uri":"uploads/dataset.csv","time_column":"time","target_columns":["s01","s02"]})
            final = runtime.execute("list_channels", {"session_id": setup["session_id"]})
        self.assertTrue(initial["ok"]); self.assertEqual(initial["result"]["structuredContent"]["summary"]["total"],0)
        self.assertTrue(converted["ok"])
        self.assertEqual(final["result"]["structuredContent"]["summary"]["total"],2)

    def test_model_catalog_search_respects_coarse_scope(self):
        models=[
            {"name":"ForecastOnly","package":"p","runtime_type":"foundation","tasks":["forecast"]},
            {"name":"AnomalyOnly","package":"p","runtime_type":"tslib","tasks":["anomaly_detection"]},
        ]
        forecast=_search_model_catalog(models,"forecast",{"session_id":"s","task_labels":["probabilistic_forecast"],"limit":10})
        anomaly=_search_model_catalog(models,"anomaly_detection",{"session_id":"s","task_labels":["anomaly_detection"],"limit":10})
        self.assertEqual([item["name"] for item in forecast["structuredContent"]["summary"]["candidates"]],["ForecastOnly"])
        self.assertEqual([item["name"] for item in anomaly["structuredContent"]["summary"]["candidates"]],["AnomalyOnly"])

    def test_final_model_names_must_come_from_this_turn_catalog_result(self):
        class CatalogClient(FakeClient):
            def complete_message(self,messages,tools=None,tool_choice=None):
                self.turn+=1
                if self.turn==1:
                    message={"role":"assistant","content":"按真实画像检索预测目录。","tool_calls":[{"id":"catalog","type":"function","function":{"name":"model_catalog_search","arguments":json.dumps({"session_id":"session_test","task_labels":["forecast"],"limit":5})}}]}
                else:
                    message={"role":"assistant","content":"目录查询真实返回了 ForecastOnly。当前只把它作为候选，并建议结合时间顺序回测、业务误差代价和部署约束再决定；现有目录记录本身不能证明未来精度。"}
                return message,{"total_tokens":4},.01,"tool_calls" if self.turn==1 else "stop"
        row=question_row(); row["task"]["model_catalog_scope"]="forecast"
        models=[{"name":"ForecastOnly","package":"p","runtime_type":"foundation","tasks":["forecast"]},{"name":"AnomalyOnly","package":"p","runtime_type":"tslib","tasks":["anomaly_detection"]}]
        record=generate_one_trajectory(row,CatalogClient(),FakeRuntime(),model_catalog=models)
        self.assertTrue(deterministic_trajectory_review(record,model_catalog=models)["passed"])
        record["final_answer"]+=" AnomalyOnly 也可以。"; record["messages"][-1]["content"]=record["final_answer"]
        review=deterministic_trajectory_review(record,model_catalog=models)
        self.assertFalse(review["passed"]); self.assertTrue(any(item.startswith("unqueried_model_name") for item in review["flags"]))

    def test_generic_transformer_is_not_a_concrete_catalog_reference(self):
        record=generate_one_trajectory(question_row(),FakeClient(),FakeRuntime())
        models=[{"name":"Transformer","tasks":["forecast"]},{"name":"Nonstationary_Transformer","tasks":["forecast"]}]
        record["final_answer"]+=" 通用的 LSTM/Transformer 架构在这里不是具体目录候选。"
        record["messages"][-1]["content"]=record["final_answer"]
        generic=deterministic_trajectory_review(record,model_catalog=models)
        self.assertTrue(generic["passed"],generic)
        self.assertTrue(any(item.startswith("generic_model_method_mentioned_without_catalog") for item in generic["warnings"]))
        record["final_answer"]+=" Nonstationary_Transformer 是具体候选。"; record["messages"][-1]["content"]=record["final_answer"]
        concrete=deterministic_trajectory_review(record,model_catalog=models)
        self.assertFalse(concrete["passed"]); self.assertIn("unqueried_model_name:Nonstationary_Transformer",concrete["flags"])

    def test_dual_candidate_competition_keeps_audit_and_exports_only_winner(self):
        base=generate_one_trajectory(question_row(),FakeClient(),FakeRuntime())
        first=copy.deepcopy(base); first["candidate_index"]=0; first["model"]="model-a"
        second=copy.deepcopy(base); second["candidate_index"]=1; second["model"]="model-b"; second["final_answer"]+=" 第二条候选补充了业务约束。"; second["messages"][-1]["content"]=second["final_answer"]
        def candidate(*args,**kwargs):
            return copy.deepcopy(first if args[2]==0 else second)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); questions=root/"questions.jsonl"; output=root/"raw.jsonl"; audit=root/"audit.jsonl"
            write_jsonl(questions,[question_row()])
            configs=[{"base_url":"https://a.test/v1","model":"model-a"},{"base_url":"https://b.test/v1","model":"model-b"}]
            with patch("api_sft.trajectories._generate_candidate",side_effect=candidate),patch("api_sft.trajectories._select_with_judge",return_value=(1,{"status":"ok","model":"judge"})) as judge:
                generate_trajectories(questions,output,audit,configs,{"model":"judge"},[],Path("/repo"),root/"ws")
            winner=json.loads(output.read_text()); competition=json.loads(audit.read_text())
        self.assertEqual(winner["model"],"model-b"); self.assertEqual(winner["competition"]["winner_index"],1)
        self.assertEqual(len(competition["candidates"]),2); self.assertEqual(competition["winner_index"],1)
        judge.assert_called_once()

    def test_competition_selects_only_hard_gate_pass_without_judge(self):
        good=generate_one_trajectory(question_row(),FakeClient(),FakeRuntime()); good["candidate_index"]=0
        bad={"id":"q1","status":"error","format_version":TRAJECTORY_FORMAT_VERSION,"question_record":question_row(),"candidate_index":1,"tool_events":[],"usage":{"total_tokens":0}}
        def candidate(*args,**kwargs): return copy.deepcopy(good if args[2]==0 else bad)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); questions=root/"questions.jsonl"; output=root/"raw.jsonl"; audit=root/"audit.jsonl"; write_jsonl(questions,[question_row()])
            with patch("api_sft.trajectories._generate_candidate",side_effect=candidate),patch("api_sft.trajectories._select_with_judge") as judge:
                generate_trajectories(questions,output,audit,[{"base_url":"https://a.test/v1","model":"a"},{"base_url":"https://b.test/v1","model":"b"}],{"model":"judge"},[],Path("/repo"),root/"ws")
            winner=json.loads(output.read_text())
        self.assertEqual(winner["competition"]["selection_rule"],"single_hard_gate_pass"); judge.assert_not_called()

    def test_competition_rejects_when_both_candidates_fail(self):
        def candidate(*args,**kwargs):
            return {"id":"q1","status":"error","format_version":TRAJECTORY_FORMAT_VERSION,"question_record":question_row(),"candidate_index":args[2],"tool_events":[],"usage":{"total_tokens":0}}
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); questions=root/"questions.jsonl"; output=root/"raw.jsonl"; audit=root/"audit.jsonl"; write_jsonl(questions,[question_row()])
            with patch("api_sft.trajectories._generate_candidate",side_effect=candidate),patch("api_sft.trajectories._select_with_judge") as judge:
                generate_trajectories(questions,output,audit,[{"base_url":"https://a.test/v1","model":"a"},{"base_url":"https://b.test/v1","model":"b"}],{"model":"judge"},[],Path("/repo"),root/"ws")
            rejected=json.loads(output.read_text()); competition=json.loads(audit.read_text())
        self.assertEqual(rejected["status"],"error"); self.assertEqual(rejected["competition"]["selection_rule"],"both_hard_gate_failed"); judge.assert_not_called()
        self.assertEqual(competition["format_version"],COMPETITION_AUDIT_FORMAT_VERSION)

    def test_selector_receives_initial_and_tool_generated_images(self):
        base=generate_one_trajectory(question_row(),FakeClient(),FakeRuntime()); first=copy.deepcopy(base); second=copy.deepcopy(base)
        first["candidate_index"]=0; second["candidate_index"]=1
        reviews=[deterministic_trajectory_review(first),deterministic_trajectory_review(second)]
        response={"candidate_scores":[{"candidate_index":0,"total_score":20},{"candidate_index":1,"total_score":21}],"winner":1,"rationale":"candidate 1"}
        with tempfile.TemporaryDirectory() as tmp:
            initial=Path(tmp)/"initial.png"; tool_image=Path(tmp)/"tool.png"; initial.write_bytes(b"png"); tool_image.write_bytes(b"png")
            row=question_row(); row["resources"]["images"]=[{"path":str(initial),"media_type":"image/png"}]; first["tool_events"][0]["image_paths"]=[str(tool_image)]
            with patch("api_sft.trajectories.OpenAICompatibleClient.complete",return_value=(json.dumps(response),{"total_tokens":3},.01)) as complete:
                winner,audit=_select_with_judge(row,[first,second],reviews,{"base_url":"https://judge.test/v1","model":"judge","api_key":"x"})
        sent=complete.call_args.args[0][1]["content"]
        self.assertEqual(winner,1); self.assertEqual(sum(block["type"]=="image_url" for block in sent),2)
        self.assertEqual(len(audit["image_manifest"]),2)


if __name__ == "__main__":
    unittest.main()
