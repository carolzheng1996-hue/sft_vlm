from __future__ import annotations

import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from api_sft.common import write_jsonl
from api_sft.trajectories import generate_one_trajectory
from api_sft.trajectories_compact import (
    COMPETITION_AUDIT_FORMAT_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    generate_trajectories,
)
from api_sft.trajectory_exporters_compact import export_trajectory_datasets
from api_sft.trajectory_verify_compact import verify_trajectories
from api_sft.tests.test_trajectories import FakeClient, FakeRuntime, question_row


def strict_loads(value: str):
    return json.loads(value, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))


class CompactTrajectoryTests(unittest.TestCase):
    def test_compact_outputs_keep_only_documented_fields_and_strict_json(self):
        base = generate_one_trajectory(question_row(), FakeClient(), FakeRuntime())
        base["tool_events"][0]["result"]["structuredContent"]["summary"]["missing_rate"] = math.nan
        base["tool_events"][0]["result_audit"].pop("full_result_sha256", None)
        good = copy.deepcopy(base)
        good.update({"candidate_index": 0, "model": "model-a", "generation_attempt": 1, "failed_attempts": []})
        failed_attempt = copy.deepcopy(base)
        failed_attempt.update({"attempt": 1, "error_type": "schema_validation", "error": "bad arguments"})
        bad = copy.deepcopy(base)
        bad.update({
            "candidate_index": 1,
            "model": "model-b",
            "status": "error",
            "generation_attempt": 1,
            "failed_attempts": [failed_attempt],
            "error_type": "schema_validation",
            "error": "bad arguments",
        })

        def candidate(*args, **kwargs):
            return copy.deepcopy(good if args[2] == 0 else bad)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            verified = root / "verified.jsonl"
            rejected = root / "rejected.jsonl"
            write_jsonl(questions, [question_row()])
            configs = [
                {"base_url": "https://a.test/v1", "model": "model-a"},
                {"base_url": "https://b.test/v1", "model": "model-b"},
            ]
            with patch("api_sft.trajectories_compact._generate_candidate", side_effect=candidate):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    configs,
                    {"model": "judge"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                )
            raw_record = strict_loads(raw.read_text())
            audit_record = strict_loads(audit.read_text())
            verify_trajectories(raw, verified, rejected)
            verified_record = strict_loads(verified.read_text())
            summary = export_trajectory_datasets(verified, root / "exports")
            trl_record = strict_loads((root / "exports" / "train_trl_tool_messages.jsonl").read_text())

        self.assertEqual(raw_record["format_version"], TRAJECTORY_FORMAT_VERSION)
        self.assertEqual(
            set(raw_record),
            {
                "id", "format_version", "question", "generation_status", "model", "candidate_index",
                "messages", "tools", "tool_calls", "final_answer", "metrics", "competition",
            },
        )
        self.assertEqual(audit_record["format_version"], COMPETITION_AUDIT_FORMAT_VERSION)
        self.assertEqual(set(audit_record), {"id", "format_version", "question", "candidates", "selection"})
        forbidden = {
            "git_dirty", "git_head", "repo_root", "workspace_path", "question_record", "tool_source",
            "execution_setup", "model_turns", "prior_attempt_errors", "internal_rubric",
        }
        serialized = json.dumps([raw_record, audit_record, verified_record], ensure_ascii=False)
        for key in forbidden:
            self.assertNotIn(f'"{key}"', serialized)
        self.assertNotIn("/private/tmp/session_test", serialized)
        self.assertIsNone(raw_record["tool_calls"][0]["result"]["missing_rate"])
        self.assertEqual(len(audit_record["candidates"][1]["failed_attempts"]), 1)
        self.assertTrue(verified_record["verification"]["passed"])
        self.assertEqual(set(trl_record), {"messages", "tools", "images"})
        self.assertEqual(summary["files"]["inspection"], "trajectories.compact.jsonl")


if __name__ == "__main__":
    unittest.main()
