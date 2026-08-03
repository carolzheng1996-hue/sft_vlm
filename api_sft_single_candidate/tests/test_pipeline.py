from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_sft.api_client import APIRequestError
from api_sft.common import write_jsonl
from api_sft.tests.test_trajectories import FakeClient, FakeRuntime, question_row
from api_sft.trajectories import generate_one_trajectory

from api_sft_single_candidate.cli import (
    _assert_resumable_formats,
    _candidate_config,
    archive_run,
)
from api_sft_single_candidate.exporters import export_trajectory_datasets
from api_sft_single_candidate.provider_adapter import (
    ProviderCompatibleClient,
    SessionBinding,
    model_tools_from_execution,
    project_tools_for_api,
)
from api_sft_single_candidate.trajectories import (
    GENERATION_AUDIT_FORMAT_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    _generate_candidate,
    generate_trajectories,
)
from api_sft_single_candidate.verify import verify_trajectories


def _strict_loads(value: str):
    return json.loads(
        value,
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


class SingleCandidatePipelineTests(unittest.TestCase):
    def _run_generation(self, root: Path, client=None, runtime=None):
        questions = root / "questions.jsonl"
        raw = root / "raw.jsonl"
        audit = root / "audit.jsonl"
        write_jsonl(questions, [question_row()])
        fake_client = client or FakeClient()
        fake_runtime = runtime or FakeRuntime()
        with (
            patch(
                "api_sft_single_candidate.trajectories.OpenAICompatibleClient",
                return_value=fake_client,
            ) as client_factory,
            patch(
                "api_sft_single_candidate.trajectories.ClaudeTsaToolRuntime",
                return_value=fake_runtime,
            ) as runtime_factory,
            patch("api_sft.trajectories._select_with_judge") as selector,
        ):
            generate_trajectories(
                questions,
                raw,
                audit,
                {
                    "base_url": "https://candidate.test/v1",
                    "model": "candidate",
                },
                [],
                Path("/repo"),
                root / "workspaces",
            )
        self.assertEqual(client_factory.call_count, 1)
        self.assertEqual(runtime_factory.call_count, 1)
        selector.assert_not_called()
        return raw, audit

    def test_one_candidate_verifies_and_exports_trl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self._run_generation(root)
            raw_record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())
            verified = root / "verified.jsonl"
            rejected = root / "rejected.jsonl"
            verify_trajectories(raw, verified, rejected)
            summary = export_trajectory_datasets(verified, root / "exports")
            trl = _strict_loads(
                (root / "exports" / "train_trl_tool_messages.jsonl").read_text()
            )

        self.assertEqual(raw_record["format_version"], TRAJECTORY_FORMAT_VERSION)
        self.assertNotIn("competition", raw_record)
        self.assertIn("generation", raw_record)
        self.assertTrue(raw_record["generation"]["hard_review"]["passed"])
        self.assertEqual(audit_record["format_version"], GENERATION_AUDIT_FORMAT_VERSION)
        self.assertEqual(set(audit_record), {"id", "format_version", "question", "candidate", "decision"})
        self.assertTrue(audit_record["decision"]["accepted"])
        self.assertEqual(set(trl), {"messages", "tools", "images"})
        self.assertEqual(summary["accepted_records"], 1)
        serialized = json.dumps(
            {"raw": raw_record, "audit": audit_record, "trl": trl},
            ensure_ascii=False,
        )
        self.assertNotIn("session_test", serialized)
        self.assertNotIn('"session_id"', serialized)

    def test_hard_review_failure_routes_to_rejected_and_keeps_audit(self):
        base = generate_one_trajectory(question_row(), FakeClient(), FakeRuntime())
        base.update(
            {
                "candidate_index": 0,
                "model": "candidate",
                "generation_attempt": 1,
                "failed_attempts": [],
            }
        )
        base["messages"][-1]["content"] = ""
        base["final_answer"] = ""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            write_jsonl(questions, [question_row()])
            with patch(
                "api_sft_single_candidate.trajectories._generate_candidate",
                return_value=base,
            ):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    {"base_url": "https://candidate.test/v1", "model": "candidate"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                )
            raw_record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())
            verified = root / "verified.jsonl"
            rejected = root / "rejected.jsonl"
            verify_trajectories(raw, verified, rejected)
            rejected_exists = rejected.exists()
            verified_exists = verified.exists()
            empty_summary = export_trajectory_datasets(verified, root / "empty_exports")
            empty_trl = (root / "empty_exports" / "train_trl_tool_messages.jsonl").read_text()

        self.assertEqual(raw_record["generation_status"], "error")
        self.assertEqual(raw_record["error"]["type"], "hard_review_failed")
        self.assertFalse(audit_record["decision"]["accepted"])
        self.assertIn(
            "empty_final_answer",
            audit_record["candidate"]["review"]["flags"],
        )
        self.assertTrue(rejected_exists)
        self.assertFalse(verified_exists)
        self.assertEqual(empty_summary["accepted_records"], 0)
        self.assertEqual(empty_trl, "")

    def test_schema_error_can_be_repaired_within_the_only_candidate(self):
        class RepairClient(FakeClient):
            def complete_message(self, messages, tools=None, tool_choice=None):
                self.turn += 1
                if self.turn == 1:
                    message = {
                        "role": "assistant",
                        "content": "先尝试画像。",
                        "tool_calls": [
                            {
                                "id": "bad",
                                "type": "function",
                                "function": {
                                    "name": "data_profile",
                                    "arguments": json.dumps(
                                        {}
                                    ),
                                },
                            }
                        ],
                    }
                elif self.turn == 2:
                    message = {
                        "role": "assistant",
                        "content": "补充缺失的数据 URI 后重试。",
                        "tool_calls": [
                            {
                                "id": "good",
                                "type": "function",
                                "function": {
                                    "name": "data_profile",
                                    "arguments": json.dumps(
                                        {
                                            "uri": "uploads/dataset.csv",
                                        }
                                    ),
                                },
                            }
                        ],
                    }
                else:
                    message = {
                        "role": "assistant",
                        "content": "修正参数后画像工具成功返回，字段和行数证据支持当前结构判断；第一次失败不作为数据结论，后续分析应继续引用成功工具结果。",
                    }
                return message, {"total_tokens": 5}, 0.01, (
                    "tool_calls" if self.turn < 3 else "stop"
                )

        with tempfile.TemporaryDirectory() as tmp:
            raw, audit = self._run_generation(Path(tmp), RepairClient())
            raw_record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())

        self.assertEqual(raw_record["generation_status"], "ok")
        self.assertEqual([event["ok"] for event in raw_record["tool_calls"]], [False, True])
        self.assertIn(
            "recovered_invalid_tool_arguments:data_profile",
            audit_record["candidate"]["review"]["warnings"],
        )

    def test_repeated_schema_failure_exhausts_attempts_and_is_rejected(self):
        class Repeater(FakeClient):
            def complete_message(self, messages, tools=None, tool_choice=None):
                self.turn += 1
                message = {
                    "role": "assistant",
                    "content": "重复错误参数。",
                    "tool_calls": [
                        {
                            "id": f"bad_{self.turn}",
                            "type": "function",
                            "function": {
                                "name": "data_profile",
                                "arguments": json.dumps(
                                    {}
                                ),
                            },
                        }
                    ],
                }
                return message, {"total_tokens": 1}, 0.01, "tool_calls"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            write_jsonl(questions, [question_row()])
            with (
                patch(
                    "api_sft_single_candidate.trajectories.OpenAICompatibleClient",
                    return_value=Repeater(),
                ),
                patch(
                    "api_sft_single_candidate.trajectories.ClaudeTsaToolRuntime",
                    return_value=FakeRuntime(),
                ) as runtime_factory,
                patch("api_sft.trajectories._select_with_judge") as selector,
            ):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    {"base_url": "https://candidate.test/v1", "model": "candidate"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                    max_attempts=2,
                )
            raw_record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())

        self.assertEqual(runtime_factory.call_count, 2)
        selector.assert_not_called()
        self.assertEqual(raw_record["generation_status"], "error")
        self.assertEqual(len(audit_record["candidate"]["failed_attempts"]), 2)
        self.assertFalse(audit_record["decision"]["accepted"])

    def test_api_boundary_failure_does_not_repeat_trajectory_attempts(self):
        class FailingClient:
            config = {"model": "candidate"}

            def __init__(self):
                self.calls = 0
                self.closed = False

            def complete_message(self, messages, tools=None, tool_choice=None):
                self.calls += 1
                raise APIRequestError(
                    "API returned HTTP 400",
                    status_code=400,
                    retryable=False,
                    attempts=1,
                    detail="invalid tool schema",
                )

            def close(self):
                self.closed = True

        client = FailingClient()
        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories.OpenAICompatibleClient",
            return_value=client,
        ), patch(
            "api_sft_single_candidate.trajectories.ClaudeTsaToolRuntime",
            return_value=FakeRuntime(),
        ) as runtime_factory:
            with self.assertRaises(APIRequestError):
                _generate_candidate(
                    question_row(),
                    {"model": "candidate"},
                    Path("/repo"),
                    Path(tmp) / "workspaces",
                    [],
                    max_attempts=2,
                    min_tool_calls=1,
                    min_distinct_tools=1,
                    max_tool_calls=8,
                    max_tool_result_chars=16000,
                )

        self.assertEqual(client.calls, 1)
        self.assertEqual(runtime_factory.call_count, 1)
        self.assertTrue(client.closed)

    def test_resume_is_idempotent_and_rejects_stale_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, audit = self._run_generation(root)
            before = raw.read_text()
            with patch(
                "api_sft_single_candidate.trajectories._generate_candidate"
            ) as candidate:
                generate_trajectories(
                    root / "questions.jsonl",
                    raw,
                    audit,
                    {
                        "base_url": "https://candidate.test/v1",
                        "model": "mock-tool-model",
                    },
                    [],
                    Path("/repo"),
                    root / "workspaces",
                    resume=True,
                )
            candidate.assert_not_called()
            self.assertEqual(raw.read_text(), before)
            stale = root / "stale.jsonl"
            write_jsonl(stale, [{"id": "q1", "format_version": "old"}])
            with self.assertRaisesRegex(RuntimeError, "use --fresh"):
                _assert_resumable_formats(
                    {
                        "raw": stale,
                        "audit": audit,
                        "verified": root / "missing_verified.jsonl",
                        "rejected": root / "missing_rejected.jsonl",
                    }
                )

    def test_fresh_archive_only_moves_single_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            single = root / "single"
            original = root / "original"
            original.mkdir()
            original_file = original / "trajectories.raw.jsonl"
            original_file.write_text('{"id":"original"}\n', encoding="utf-8")
            p = {
                "raw": single / "trajectories.raw.jsonl",
                "audit": single / "trajectories.generation_audit.jsonl",
                "verified": single / "trajectories.verified.jsonl",
                "rejected": single / "trajectories.rejected.jsonl",
                "workspaces": single / "trajectory_workspaces",
                "exports": single / "trajectory_exports",
                "archives": single / "trajectory_run_archives",
            }
            write_jsonl(p["raw"], [{"id": "q1", "format_version": TRAJECTORY_FORMAT_VERSION}])
            write_jsonl(
                p["audit"],
                [{"id": "q1", "format_version": GENERATION_AUDIT_FORMAT_VERSION}],
            )
            archive = archive_run(p)
            original_exists = original_file.exists()
            original_content = original_file.read_text()
            raw_exists = p["raw"].exists()

        self.assertIsNotNone(archive)
        self.assertTrue(original_exists)
        self.assertEqual(original_content, '{"id":"original"}\n')
        self.assertFalse(raw_exists)

    def test_strict_json_converts_non_finite_values(self):
        candidate = generate_one_trajectory(question_row(), FakeClient(), FakeRuntime())
        candidate["tool_events"][0]["result"]["structuredContent"]["summary"]["nan"] = math.nan
        candidate["tool_events"][0]["result_audit"].pop("full_result_sha256", None)
        candidate.update(
            {
                "candidate_index": 0,
                "model": "candidate",
                "generation_attempt": 1,
                "failed_attempts": [],
            }
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            write_jsonl(questions, [question_row()])
            with patch(
                "api_sft_single_candidate.trajectories._generate_candidate",
                return_value=copy.deepcopy(candidate),
            ):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    {"base_url": "https://candidate.test/v1", "model": "candidate"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                )
            raw_record = _strict_loads(raw.read_text())

        self.assertIsNone(raw_record["tool_calls"][0]["result"]["nan"])

    def test_export_preserves_initial_and_tool_generated_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, _ = self._run_generation(root)
            record = _strict_loads(raw.read_text())
            record["messages"][1]["images"] = ["/tmp/initial.png"]
            record["messages"][3]["images"] = ["/tmp/tool_plot.png"]
            verified = root / "verified.jsonl"
            write_jsonl(verified, [record])
            export_trajectory_datasets(verified, root / "exports")
            trl = _strict_loads(
                (root / "exports" / "train_trl_tool_messages.jsonl").read_text()
            )

        self.assertEqual(trl["images"], ["/tmp/initial.png", "/tmp/tool_plot.png"])
        self.assertEqual(trl["messages"][1]["content"][0], {"type": "image"})
        self.assertEqual(trl["messages"][4]["role"], "user")
        self.assertEqual(trl["messages"][4]["content"][0], {"type": "image"})

    def test_config_accepts_one_candidate_and_does_not_require_selector(self):
        config = {
            "trajectory_generation": {"timeout_seconds": 9, "retries": 2},
            "models": {
                "candidate": {
                    "base_url": "https://candidate.test/v1",
                    "model": "tool-model",
                    "api_key": "test-secret",
                }
            },
        }
        candidate = _candidate_config(config)
        self.assertEqual(candidate["timeout_seconds"], 9)
        self.assertEqual(candidate["retries"], 2)
        self.assertEqual(candidate["api_key"], "test-secret")
        self.assertFalse(candidate["response_format"])
        self.assertTrue(candidate["_reuse_http_client"])
        self.assertTrue(candidate["_tools_preprojected"])
        with self.assertRaisesRegex(RuntimeError, "models.candidate"):
            _candidate_config({"models": {"trajectory_selector": {"model": "judge"}}})
        invalid = copy.deepcopy(config)
        invalid["models"]["candidate"]["provider"] = "unknown"
        with self.assertRaisesRegex(RuntimeError, "provider"):
            _candidate_config(invalid)
        missing_key = copy.deepcopy(config)
        missing_key["models"]["candidate"].pop("api_key")
        with self.assertRaisesRegex(RuntimeError, "api_key"):
            _candidate_config(missing_key)
        placeholder = copy.deepcopy(config)
        placeholder["models"]["candidate"]["api_key"] = "replace-with-your-api-key"
        with self.assertRaisesRegex(RuntimeError, "placeholder") as caught:
            _candidate_config(placeholder)
        self.assertNotIn("replace-with-your-api-key", str(caught.exception))

    def test_runtime_injects_session_without_exposing_it_to_model(self):
        class CapturingClient(FakeClient):
            def __init__(self):
                super().__init__()
                self.requests = []

            def complete_message(self, messages, tools=None, tool_choice=None):
                self.requests.append(copy.deepcopy({"messages": messages, "tools": tools}))
                self.turn += 1
                if self.turn == 1:
                    return {
                        "role": "assistant",
                        "content": "读取数据画像。",
                        "tool_calls": [{
                            "id": "profile",
                            "type": "function",
                            "function": {
                                "name": "data_profile",
                                "arguments": json.dumps({
                                    "session_id": "malicious_session",
                                    "uri": "uploads/dataset.csv",
                                }),
                            },
                        }],
                    }, {"total_tokens": 1}, 0.01, "tool_calls"
                return {
                    "role": "assistant",
                    "content": "画像结果提供了真实字段与行数证据，足以完成当前结构判断；后续建模仍应采用时间顺序验证并依据业务误差代价选择指标。",
                }, {"total_tokens": 1}, 0.01, "stop"

        class TrackingRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.executed = []

            def execute(self, name, arguments):
                self.executed.append(copy.deepcopy(arguments))
                return super().execute(name, arguments)

        with tempfile.TemporaryDirectory() as tmp:
            client = CapturingClient()
            runtime = TrackingRuntime()
            raw, _ = self._run_generation(Path(tmp), client, runtime)
            record = _strict_loads(raw.read_text())

        request_json = json.dumps(client.requests, ensure_ascii=False)
        record_json = json.dumps(record, ensure_ascii=False)
        self.assertNotIn("session_test", request_json)
        self.assertNotIn('"session_id"', request_json)
        self.assertEqual(runtime.executed[0]["session_id"], "session_test")
        self.assertNotEqual(runtime.executed[0]["session_id"], "malicious_session")
        self.assertNotIn("session_test", record_json)
        self.assertNotIn("malicious_session", record_json)
        self.assertNotIn('"session_id"', record_json)

    def test_adapter_only_decodes_valid_json_containers(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "data_convert",
                "description": "convert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "const": "session_test"},
                        "target_columns": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "minItems": 1,
                        },
                    },
                    "required": ["session_id", "target_columns"],
                },
            },
        }]

        class Inner:
            config = {"model": "qwen"}

            def __init__(self, value):
                self.value = value
                self.request_tools = None

            def complete_message(self, messages, tools=None, tool_choice=None):
                self.request_tools = copy.deepcopy(tools)
                return {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "convert",
                        "type": "function",
                        "function": {
                            "name": "data_convert",
                            "arguments": json.dumps({"target_columns": self.value}),
                        },
                    }],
                }, {"total_tokens": 1}, 0.01, "tool_calls"

        valid_inner = Inner('["s01","s02"]')
        adapter = ProviderCompatibleClient(
            valid_inner,
            SessionBinding("session_test"),
        )
        message, _, _, _ = adapter.complete_message([], tools=tools, tool_choice="auto")
        arguments = json.loads(message["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(arguments["target_columns"], ["s01", "s02"])
        self.assertEqual(arguments["session_id"], "session_test")
        self.assertEqual(adapter.audit_summary()["argument_json_decoded"], 1)
        self.assertNotIn(
            "session_id",
            valid_inner.request_tools[0]["function"]["parameters"]["properties"],
        )

        invalid_inner = Inner("s01,s02")
        invalid_adapter = ProviderCompatibleClient(
            invalid_inner,
            SessionBinding("session_test"),
        )
        invalid_message, _, _, _ = invalid_adapter.complete_message(
            [], tools=tools, tool_choice="auto"
        )
        invalid_arguments = json.loads(
            invalid_message["tool_calls"][0]["function"]["arguments"]
        )
        self.assertEqual(invalid_arguments["target_columns"], "s01,s02")
        self.assertEqual(invalid_adapter.audit_summary()["argument_json_decoded"], 0)

    def test_portable_projection_does_not_mutate_model_schema(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "data_convert",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "const": "session_test"},
                        "target_columns": {
                            "type": ["array", "null"],
                            "items": {"type": "string"},
                            "default": None,
                        },
                        "language": {
                            "type": ["string", "null"],
                            "enum": ["en", "zh-CN", None],
                        },
                    },
                    "required": ["session_id"],
                    "anyOf": [{"required": ["target_columns"]}],
                    "additionalProperties": False,
                },
            },
        }]
        original = copy.deepcopy(tools)
        model_tools = model_tools_from_execution(tools)
        projected_tools = project_tools_for_api(model_tools)

        self.assertEqual(tools, original)
        self.assertIn("anyOf", model_tools[0]["function"]["parameters"])
        schema = projected_tools[0]["function"]["parameters"]
        self.assertNotIn("anyOf", schema)
        self.assertNotIn("additionalProperties", schema)
        self.assertEqual(
            schema["properties"]["target_columns"]["type"],
            "array",
        )
        self.assertNotIn("default", schema["properties"]["target_columns"])
        self.assertEqual(schema["properties"]["language"]["type"], "string")
        self.assertEqual(schema["properties"]["language"]["enum"], ["en", "zh-CN"])

    @unittest.skipUnless(
        sys.version_info >= (3, 11)
        and Path("/Users/monychen/Documents/tsa/claude_tsa/pulsar/tsa/tools").is_dir(),
        "live claude_tsa schemas require Python 3.11+",
    )
    def test_all_live_tool_schemas_project_without_runtime_fields(self):
        from api_sft.tool_runtime import ClaudeTsaToolRuntime

        with tempfile.TemporaryDirectory() as tmp:
            runtime = ClaudeTsaToolRuntime(
                Path("/Users/monychen/Documents/tsa/claude_tsa"),
                Path(tmp),
            )
            described = runtime.describe_tools(session_id="session_test")
        execution_tools = [
            {
                "type": "function",
                "function": {
                    "name": item["name"],
                    "description": item["description"],
                    "parameters": item["input_schema"],
                },
            }
            for item in described
        ]
        original = copy.deepcopy(execution_tools)
        model_tools = model_tools_from_execution(execution_tools)

        def walk(value):
            if isinstance(value, dict):
                yield value
                for child in value.values():
                    yield from walk(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)

        self.assertGreater(len(model_tools), 20)
        self.assertEqual(execution_tools, original)
        for tool in model_tools:
            schema = tool["function"]["parameters"]
            self.assertNotIn("session_id", schema.get("properties", {}))
            self.assertNotIn("session_id", schema.get("required", []))
        projected = project_tools_for_api(model_tools)
        for node in walk(projected):
            self.assertFalse(
                {"anyOf", "oneOf", "allOf", "$ref", "$defs"} & set(node)
            )
            if isinstance(node.get("type"), list):
                self.assertNotIn("null", node["type"])
            if isinstance(node.get("enum"), list) and node.get("type") == "string":
                self.assertNotIn(None, node["enum"])

    def test_multiple_tool_calls_are_retried_once_without_execution(self):
        class MultiThenSingle(FakeClient):
            def complete_message(self, messages, tools=None, tool_choice=None):
                self.turn += 1
                call = lambda call_id: {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "data_profile",
                        "arguments": json.dumps({"uri": "uploads/dataset.csv"}),
                    },
                }
                if self.turn == 1:
                    return {
                        "role": "assistant",
                        "content": "并行尝试。",
                        "tool_calls": [call("first"), call("second")],
                    }, {"total_tokens": 1}, 0.01, "tool_calls"
                if self.turn == 2:
                    return {
                        "role": "assistant",
                        "content": "只保留必要的画像调用。",
                        "tool_calls": [call("profile")],
                    }, {"total_tokens": 1}, 0.01, "tool_calls"
                return {
                    "role": "assistant",
                    "content": "单个画像调用返回了真实结构证据，当前可以据此完成判断；未执行的并行建议不构成证据，后续分析仍应按工具结果逐步推进。",
                }, {"total_tokens": 1}, 0.01, "stop"

        class TrackingRuntime(FakeRuntime):
            def __init__(self):
                super().__init__()
                self.executions = 0

            def execute(self, name, arguments):
                self.executions += 1
                return super().execute(name, arguments)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = TrackingRuntime()
            raw, audit = self._run_generation(
                Path(tmp),
                MultiThenSingle(),
                runtime,
            )
            record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())

        self.assertEqual(runtime.executions, 1)
        self.assertEqual(len(record["tool_calls"]), 1)
        adapter = audit_record["candidate"]["provider_adapter"]
        self.assertEqual(adapter["multi_call_retried"], 1)
        self.assertIsNone(adapter["adapter_error"])

    def test_persistent_multiple_calls_fail_the_attempt(self):
        class AlwaysMultiple(FakeClient):
            def complete_message(self, messages, tools=None, tool_choice=None):
                self.turn += 1
                calls = [
                    {
                        "id": f"call_{self.turn}_{index}",
                        "type": "function",
                        "function": {
                            "name": "data_profile",
                            "arguments": json.dumps({"uri": "uploads/dataset.csv"}),
                        },
                    }
                    for index in range(2)
                ]
                return {
                    "role": "assistant",
                    "content": "仍然并行调用。",
                    "tool_calls": calls,
                }, {"total_tokens": 1}, 0.01, "tool_calls"

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            write_jsonl(questions, [question_row()])
            runtime = FakeRuntime()
            with (
                patch(
                    "api_sft_single_candidate.trajectories.OpenAICompatibleClient",
                    return_value=AlwaysMultiple(),
                ),
                patch(
                    "api_sft_single_candidate.trajectories.ClaudeTsaToolRuntime",
                    return_value=runtime,
                ),
            ):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    {
                        "base_url": "https://candidate.test/v1",
                        "model": "candidate",
                    },
                    [],
                    Path("/repo"),
                    root / "workspaces",
                    max_attempts=1,
                )
            record = _strict_loads(raw.read_text())
            audit_record = _strict_loads(audit.read_text())

        self.assertEqual(record["generation_status"], "error")
        failed_adapter = audit_record["candidate"]["failed_attempts"][0][
            "provider_adapter"
        ]
        self.assertEqual(failed_adapter["multi_call_retried"], 1)
        self.assertEqual(
            failed_adapter["adapter_error"],
            "provider_returned_multiple_tool_calls_after_retry",
        )


if __name__ == "__main__":
    unittest.main()
