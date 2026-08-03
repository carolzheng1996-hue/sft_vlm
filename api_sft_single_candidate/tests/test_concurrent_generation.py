from __future__ import annotations

import copy
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from api_sft.api_client import APIRequestError
from api_sft.common import iter_jsonl, write_jsonl
from api_sft.tests.test_trajectories import question_row
from api_sft.tests.test_trajectories import FakeClient, FakeRuntime

from api_sft_single_candidate.trajectories import (
    GENERATION_AUDIT_FORMAT_VERSION,
    TRAJECTORY_FORMAT_VERSION,
    _generate_candidate,
    generate_trajectories,
)


def _rows(count: int) -> list[dict]:
    result = []
    for index in range(count):
        row = copy.deepcopy(question_row())
        row["id"] = f"q{index:02d}"
        result.append(row)
    return result


def _fake_result(row: dict, accepted: bool = True) -> dict:
    return {
        "id": row["id"],
        "audit": {
            "id": row["id"],
            "format_version": GENERATION_AUDIT_FORMAT_VERSION,
            "decision": {"accepted": accepted},
        },
        "output": {
            "id": row["id"],
            "format_version": TRAJECTORY_FORMAT_VERSION,
            "generation_status": "ok" if accepted else "error",
            "model": "candidate",
        },
        "accepted": accepted,
    }


class ConcurrentGenerationTests(unittest.TestCase):
    def _run(
        self,
        root: Path,
        rows: list[dict],
        max_concurrency: int = 4,
        resume: bool = False,
    ) -> tuple[Path, Path]:
        questions = root / "questions.jsonl"
        raw = root / "raw.jsonl"
        audit = root / "audit.jsonl"
        write_jsonl(questions, rows)
        generate_trajectories(
            questions,
            raw,
            audit,
            {"model": "candidate"},
            [],
            Path("/repo"),
            root / "workspaces",
            resume=resume,
            max_concurrency=max_concurrency,
        )
        return raw, audit

    def test_dynamic_pool_is_bounded_and_main_thread_writes_each_record(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(row, *args):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.015 if row["id"] != "q00" else 0.06)
            with lock:
                active -= 1
            return _fake_result(row)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            raw, audit = self._run(Path(tmp), _rows(9), max_concurrency=4)
            raw_rows = list(iter_jsonl(raw))
            audit_rows = list(iter_jsonl(audit))

        self.assertGreater(maximum, 1)
        self.assertLessEqual(maximum, 4)
        self.assertEqual({row["id"] for row in raw_rows}, {f"q{i:02d}" for i in range(9)})
        self.assertEqual({row["id"] for row in audit_rows}, {f"q{i:02d}" for i in range(9)})
        self.assertEqual(len(raw_rows), 9)
        self.assertEqual(len(audit_rows), 9)

    def test_canary_is_written_once_and_quality_rejection_does_not_stop_pool(self):
        calls: list[str] = []

        def worker(row, *args):
            calls.append(row["id"])
            return _fake_result(row, accepted=row["id"] != "q00")

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            raw, _ = self._run(Path(tmp), _rows(4), max_concurrency=2)
            records = list(iter_jsonl(raw))

        self.assertEqual(calls.count("q00"), 1)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["id"], "q00")
        self.assertEqual(records[0]["generation_status"], "error")

    def test_global_api_error_on_canary_stops_before_pool_submission(self):
        calls: list[str] = []

        def worker(row, *args):
            calls.append(row["id"])
            raise APIRequestError(
                "API returned HTTP 400",
                status_code=400,
                retryable=False,
                attempts=1,
                detail="invalid tool schema",
            )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "canary"):
                self._run(root, _rows(5), max_concurrency=4)
            self.assertFalse((root / "raw.jsonl").exists())
            self.assertFalse((root / "audit.jsonl").exists())

        self.assertEqual(calls, ["q00"])

    def test_global_api_error_in_pool_stops_new_submissions_and_keeps_checkpoints(self):
        calls: list[str] = []

        def worker(row, *args):
            calls.append(row["id"])
            if row["id"] == "q01":
                raise APIRequestError(
                    "API transport failed: ConnectError",
                    status_code=None,
                    retryable=True,
                    attempts=3,
                    detail="connection closed",
                )
            if row["id"] == "q02":
                time.sleep(0.01)
            return _fake_result(row)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, "stopped further"):
                self._run(root, _rows(6), max_concurrency=2)
            completed = {row["id"] for row in iter_jsonl(root / "raw.jsonl")}

        self.assertEqual(completed, {"q00", "q02"})
        self.assertEqual(set(calls), {"q00", "q01", "q02"})

    def test_worker_exception_isolated_as_rejected_record(self):
        def worker(row, *args):
            if row["id"] == "q02":
                raise TimeoutError("worker timeout")
            return _fake_result(row)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            raw, audit = self._run(Path(tmp), _rows(5), max_concurrency=3)
            raw_rows = {row["id"]: row for row in iter_jsonl(raw)}
            audit_rows = {row["id"]: row for row in iter_jsonl(audit)}

        self.assertEqual(len(raw_rows), 5)
        self.assertEqual(len(audit_rows), 5)
        self.assertEqual(raw_rows["q02"]["generation_status"], "error")
        self.assertEqual(audit_rows["q02"]["decision"]["accepted"], False)
        self.assertEqual(raw_rows["q01"]["generation_status"], "ok")

    def test_resume_repairs_orphan_record_and_only_reprocesses_missing_id(self):
        calls: list[str] = []

        def worker(row, *args):
            calls.append(row["id"])
            return _fake_result(row)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            root = Path(tmp)
            raw, audit = self._run(root, _rows(3), max_concurrency=2)
            original_audit = list(iter_jsonl(audit))
            audit.write_text(
                "\n".join(
                    json.dumps(row, ensure_ascii=False)
                    for row in original_audit
                    if row["id"] != "q02"
                )
                + "\n",
                encoding="utf-8",
            )
            calls.clear()
            self._run(root, _rows(3), max_concurrency=2, resume=True)
            raw_rows = list(iter_jsonl(raw))
            audit_rows = list(iter_jsonl(audit))

        self.assertEqual(calls, ["q02"])
        self.assertEqual(len(raw_rows), 3)
        self.assertEqual(len(audit_rows), 3)

    def test_resume_rejects_records_from_a_different_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            raw = root / "raw.jsonl"
            audit = root / "audit.jsonl"
            write_jsonl(questions, _rows(1))
            write_jsonl(
                raw,
                [{
                    "id": "q00",
                    "format_version": TRAJECTORY_FORMAT_VERSION,
                    "generation_status": "ok",
                    "model": "old-model",
                }],
            )
            write_jsonl(
                audit,
                [{
                    "id": "q00",
                    "format_version": GENERATION_AUDIT_FORMAT_VERSION,
                    "decision": {"accepted": True},
                }],
            )
            with self.assertRaisesRegex(RuntimeError, "different model"):
                generate_trajectories(
                    questions,
                    raw,
                    audit,
                    {"model": "new-model"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                    resume=True,
                )

    def test_serial_mode_keeps_single_in_flight_worker(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        def worker(row, *args):
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.005)
            with lock:
                active -= 1
            return _fake_result(row)

        with tempfile.TemporaryDirectory() as tmp, patch(
            "api_sft_single_candidate.trajectories._process_row",
            side_effect=worker,
        ):
            self._run(Path(tmp), _rows(4), max_concurrency=1)

        self.assertEqual(maximum, 1)

    def test_concurrency_must_be_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            questions = root / "questions.jsonl"
            write_jsonl(questions, _rows(1))
            with self.assertRaisesRegex(ValueError, "at least 1"):
                generate_trajectories(
                    questions,
                    root / "raw.jsonl",
                    root / "audit.jsonl",
                    {"model": "candidate"},
                    [],
                    Path("/repo"),
                    root / "workspaces",
                    max_concurrency=0,
                )

    def test_each_question_gets_a_distinct_runtime_workspace(self):
        runtime_paths: list[Path] = []

        def runtime_factory(repo_root, workspace):
            runtime_paths.append(workspace)
            return FakeRuntime()

        with patch(
            "api_sft_single_candidate.trajectories.OpenAICompatibleClient",
            side_effect=lambda config: FakeClient(),
        ), patch(
            "api_sft_single_candidate.trajectories.ClaudeTsaToolRuntime",
            side_effect=runtime_factory,
        ):
            for row in _rows(2):
                _generate_candidate(
                    row,
                    {"model": "candidate"},
                    Path("/repo"),
                    Path("/tmp/workspaces"),
                    [],
                    max_attempts=1,
                    min_tool_calls=1,
                    min_distinct_tools=1,
                    max_tool_calls=8,
                    max_tool_result_chars=16000,
                )

        self.assertEqual(len(runtime_paths), 2)
        self.assertNotEqual(runtime_paths[0], runtime_paths[1])
        self.assertTrue(all(path.name == "attempt_01" for path in runtime_paths))


if __name__ == "__main__":
    unittest.main()
