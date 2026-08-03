from __future__ import annotations

import json
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from api_sft.common import iter_jsonl, stable_hash, write_jsonl
from api_sft.questions_concurrent import _groups_in_input_order, generate_questions_concurrent


ROOT = Path(__file__).resolve().parents[2]


class FakeWriterClient:
    active = 0
    max_active = 0
    calls = 0
    lock = threading.Lock()

    def __init__(self, config: dict):
        self.config = config

    @classmethod
    def reset(cls) -> None:
        with cls.lock:
            cls.active = 0
            cls.max_active = 0
            cls.calls = 0

    def complete(self, messages):
        with self.lock:
            type(self).calls += 1
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)
            call_number = type(self).calls
        try:
            time.sleep(0.03)
            response = {
                "user_request": f"问题请求 {call_number}：请检查证据并决定下一步分析。",
                "decision_points": ["检查证据", "决定下一步"],
                "constraint_key": "compute_budget",
                "business_facts_used": ["decision_constraints.compute_budget"],
            }
            return json.dumps(response, ensure_ascii=False), {"total_tokens": 12}, 0.03
        finally:
            with self.lock:
                type(self).active -= 1


class ConcurrentQuestionTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeWriterClient.reset()

    def _spec(self, row_id: str, mode: str, group_id: str) -> dict:
        pair_id = group_id if group_id.startswith("pair") else None
        return {
            "id": row_id,
            "question_spec_version": "4.0",
            "question_spec_hash": stable_hash({"id": row_id, "mode": mode, "scenario": "hash"}),
            "question_group_id": group_id,
            "scenario_id": f"scenario_{group_id}",
            "pair_id": pair_id,
            "is_paired": bool(pair_id),
            "pair_role": mode if pair_id else f"standalone_{mode}",
            "split_group": f"scenario_{group_id}",
            "task": "data_profile",
            "task_label": "数据画像",
            "subtask_id": "profile_schema_frequency_index",
            "subtask_title": "字段、频率与时间索引质量检查",
            "task_goal": "forecast",
            "series_count": "single",
            "history_length": "long",
            "difficulty": "medium",
            "input_mode": mode,
            "data_path": "/tmp/data.csv",
            "dataset_attachment": {"path": "/tmp/data.csv", "format": "csv", "source_type": "local_path"},
            "images": ["/tmp/x.png"] if mode == "image_text" else [],
            "image_inventory": ["x.png"] if mode == "image_text" else [],
            "image_attachments": [],
            "evidence_packet": {
                "schema": {"columns": ["time", "s01"]},
                "data_scale": {"row_count": 803, "series_count": 1, "history_length_per_series": 803},
                "time_index": {"frequency": "synthetic_step", "observed_range": {"start": "0", "end": "802"}},
                "statistics": {"s01": {"mean": 1.2}},
                "business_constraints": {"compute_budget": "low", "interpretability": "required", "error_cost": "under_forecast_higher"},
                "known_future_covariates": [],
            },
            "visible_context": {},
            "recommended_mode": "text_only",
            "image_value": "low",
            "visual_reason": "x",
            "recommended_plots": [],
            "text_can_answer": [],
            "image_should_answer": [],
            "requires_statistical_confirmation": [],
            "model_catalog_scope": "none",
            "system_prompt": "TOOL SYSTEM",
            "message_format": "neutral_local_images_v1",
            "trajectory_requirement": "tool_execution",
            "candidate_tools": [],
            "primary_tools": ["data_profile"],
            "required_answer_elements": ["schema判断"],
            "internal_rubric": {
                "task_instruction": "internal-only",
                "required_elements": ["schema判断"],
                "preferred_tools": ["data_profile"],
                "image_policy": "low",
                "expected_decision_points": 2,
                "allowed_user_tool_mentions": [],
                "evidence_boundaries": [],
            },
            "scenario_hash": "hash",
        }

    def _run(self, root: Path, specs: list[dict], **kwargs):
        specs_path = root / "question_specs.jsonl"
        write_jsonl(specs_path, specs)
        prompt_path = root / "writer.txt"
        prompt_path.write_text("只输出 JSON", encoding="utf-8")
        return generate_questions_concurrent(
            specs_path,
            root / "concurrent_questions",
            {"base_url": "https://example.invalid/v1", "model": "writer", "api_key": "secret"},
            prompt_path,
            **kwargs,
        )

    def test_group_limit_preserves_spec_order_instead_of_pair_prefix_sorting(self):
        rows = [
            {"id": "pair_a_text", "question_group_id": "pair_a"},
            {"id": "pair_a_image", "question_group_id": "pair_a"},
            {"id": "scenario_00002_q01", "question_group_id": "scenario_00002_q01"},
            {"id": "scenario_00003_q01", "question_group_id": "scenario_00003_q01"},
        ]
        groups = _groups_in_input_order(rows)
        self.assertEqual(
            [group[0]["question_group_id"] for group in groups],
            ["pair_a", "scenario_00002_q01", "scenario_00003_q01"],
        )

    def test_four_groups_are_concurrent_and_paired_group_uses_one_call(self):
        specs = [
            self._spec("q1", "text_only", "g1"),
            self._spec("q2", "text_only", "g2"),
            self._spec("q3", "image_text", "g3"),
            self._spec("q4", "text_only", "g4"),
            self._spec("q5", "text_only", "pair_g5"),
            self._spec("q6", "image_text", "pair_g5"),
        ]
        with TemporaryDirectory() as tmp, patch(
            "api_sft.questions_concurrent.OpenAICompatibleClient", FakeWriterClient
        ), patch(
            "api_sft.questions_concurrent.validate_question",
            return_value={"passed": True, "checks": {}, "details": {}},
        ), patch(
            "api_sft.questions_concurrent.validate_question_diversity",
            return_value={"passed": True, "checks": {}},
        ):
            rows = self._run(Path(tmp), specs, max_concurrency=4)
            output = Path(tmp) / "concurrent_questions" / "questions.final.jsonl"
            generated = list(iter_jsonl(output))

        self.assertEqual(FakeWriterClient.calls, 5)
        self.assertGreaterEqual(FakeWriterClient.max_active, 2)
        self.assertLessEqual(FakeWriterClient.max_active, 4)
        self.assertEqual(len(rows), 6)
        self.assertEqual(len(generated), 6)
        pair = [row for row in generated if row["id"] in {"q5", "q6"}]
        self.assertEqual(pair[0]["prompt"]["user_request"], pair[1]["prompt"]["user_request"])
        self.assertEqual(
            set(generated[0]),
            {"id", "format_version", "spec_ref", "task", "prompt", "resources", "allowed_tools"},
        )
        self.assertEqual(generated[0]["format_version"], "question_runtime_v2")
        self.assertEqual([row["id"] for row in generated], sorted(row["id"] for row in generated))

    def test_resume_does_not_repeat_completed_groups_and_old_output_is_untouched(self):
        specs = [self._spec("q1", "text_only", "g1"), self._spec("q2", "text_only", "g2")]
        with TemporaryDirectory() as tmp, patch(
            "api_sft.questions_concurrent.OpenAICompatibleClient", FakeWriterClient
        ), patch(
            "api_sft.questions_concurrent.validate_question",
            return_value={"passed": True, "checks": {}, "details": {}},
        ), patch(
            "api_sft.questions_concurrent.validate_question_diversity",
            return_value={"passed": True, "checks": {}},
        ):
            root = Path(tmp)
            old = root / "questions.final.jsonl"
            old.write_text("old-output\n", encoding="utf-8")
            self._run(root, specs, max_concurrency=2)
            first_calls = FakeWriterClient.calls
            self._run(root, specs, max_concurrency=2, resume=True)
            old_contents = old.read_text(encoding="utf-8")

        self.assertEqual(first_calls, 2)
        self.assertEqual(FakeWriterClient.calls, first_calls)
        self.assertEqual(old_contents, "old-output\n")

    def test_fresh_archives_only_concurrent_outputs(self):
        specs = [self._spec("q1", "text_only", "g1")]
        with TemporaryDirectory() as tmp, patch(
            "api_sft.questions_concurrent.OpenAICompatibleClient", FakeWriterClient
        ), patch(
            "api_sft.questions_concurrent.validate_question",
            return_value={"passed": True, "checks": {}, "details": {}},
        ), patch(
            "api_sft.questions_concurrent.validate_question_diversity",
            return_value={"passed": True, "checks": {}},
        ):
            root = Path(tmp)
            self._run(root, specs, max_concurrency=1)
            self._run(root, specs, max_concurrency=1, fresh=True)
            archives = list((root / "concurrent_questions" / "archives").glob("*/questions.final.jsonl"))

        self.assertEqual(len(archives), 1)


if __name__ == "__main__":
    unittest.main()
