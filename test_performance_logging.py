"""Offline performance instrumentation checks; no network calls."""

import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import rag


class PerformanceLoggingTests(unittest.TestCase):
    def test_timer_uses_perf_counter_and_logs_on_error(self):
        with self.assertLogs(rag._performance_log, level="INFO") as logs:
            with patch("rag.perf_counter", side_effect=[10.0, 10.125]):
                with self.assertRaises(ValueError):
                    with rag.performance_timer("test_stage"):
                        raise ValueError("private text")
        self.assertIn("elapsed_seconds=0.125000", logs.output[0])
        self.assertNotIn("private text", logs.output[0])

    def test_separate_stages_with_mocked_model_calls(self):
        sources = [dict(id=1, filename="private.pdf", page=1,
                        text="3703 #4\nSPR\nPrivate function description.\n0: Not performed\n1: Performed")]
        response = dict(supported=True, answer="İşlevi belirler.",
                        evidence=[dict(source_id=1, quote="Private function description.")])
        with self.assertLogs(rag._performance_log, level="INFO") as logs:
            with tempfile.TemporaryDirectory() as folder:
                from pathlib import Path
                self.assertEqual(rag.retrieve_context("Explain 3703", Path(folder)), [])
            with patch("rag.retrieve_context", return_value=sources), patch(
                    "rag.local_chat", side_effect=[response, {"verified": True}]), contextlib.redirect_stdout(io.StringIO()):
                rag.answer_question("Explain 3703#4 SPR")
        output = "\n".join(logs.output)
        for stage in ("retrieval", "parameter_extraction", "qwen_answer", "qwen_verification", "total_request"):
            self.assertIn("stage=" + stage, output)
        self.assertNotIn("Private function", output)
        self.assertNotIn("private.pdf", output)

    def test_metadata_is_optional_and_never_logs_response_text(self):
        for metrics in ({}, dict(total_duration=1000, load_duration=200,
                                 prompt_eval_count=30, eval_count=12)):
            response = dict(metrics, message={"content": json.dumps({"answer": "private answer"})})
            with patch("rag.request.build_opener") as build:
                build.return_value.open.return_value.__enter__.return_value = io.StringIO(json.dumps(response))
                with self.assertLogs(rag._performance_log, level="INFO") as logs:
                    self.assertEqual(rag.local_chat({}, stage="answer"), {"answer": "private answer"})
            output = "\n".join(logs.output)
            self.assertNotIn("private answer", output)
            for key, value in metrics.items():
                self.assertIn(f"'{key}': {value}", output)


if __name__ == "__main__":
    unittest.main()
