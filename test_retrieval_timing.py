"""Offline retrieval timing tests with fake PDFs and query embeddings."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import hybrid_search
import rag


class RetrievalTimingTests(unittest.TestCase):
    def test_assistant_times_real_scan_operations_without_hybrid(self):
        page = MagicMock()
        page.get_text.return_value = "3703 #4 SPR\nPrivate PDF description.\n0: Not performed\n1: Performed"
        pdf = MagicMock()
        pdf.needs_pass = False
        pdf.__iter__.return_value = iter([page])
        pdf.__len__.return_value = 1
        pdf.__enter__.return_value = pdf
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "private.pdf").touch()
            with patch("rag.pymupdf.open", return_value=pdf), patch(
                    "hybrid_search.search") as hybrid, self.assertLogs(rag._performance_log) as logs:
                results = rag.retrieve_context("Explain 3703#4 SPR", Path(folder))
        self.assertEqual(len(results), 1)
        hybrid.assert_not_called()
        output = "\n".join(logs.output)
        for stage in ("pdf_open", "pdf_read", "keyword_search", "label_scoring", "context_selection"):
            self.assertIn("stage=retrieval." + stage, output)
        self.assertIn("reason=hybrid_search_not_used", output)
        self.assertNotIn("stage=hybrid.", output)
        self.assertNotIn("pdf_read_continuation", output)
        self.assertNotIn("Private PDF description", output)
        self.assertNotIn("private.pdf", output)

    def test_hybrid_stages_run_only_with_cached_chunks(self):
        chunks = [dict(filename="private.pdf", page=1, text="Private spindle description")]
        with patch("hybrid_search.load_cached_chunks", return_value=(chunks, [[1, 0]])), patch(
                "hybrid_search.semantic.embed_texts", return_value=[[1, 0]]) as embed:
            with self.assertLogs(rag._performance_log) as logs:
                results = hybrid_search.search("spindle")
        embed.assert_called_once_with(["spindle"])
        self.assertEqual(results[0]["chunk_id"], 0)
        output = "\n".join(logs.output)
        for stage in ("cache_loading", "query_embedding", "semantic_scoring", "keyword_search", "rank_fusion"):
            self.assertIn("stage=hybrid." + stage, output)
        self.assertIn("reason=cached_chunks_only", output)
        self.assertNotIn("Private spindle description", output)
        with patch("hybrid_search.load_cached_chunks", return_value=([], [])), patch(
                "hybrid_search.semantic.embed_texts") as embed:
            with self.assertLogs(rag._performance_log) as logs:
                self.assertEqual(hybrid_search.search("spindle"), [])
        embed.assert_not_called()
        output = "\n".join(logs.output)
        self.assertIn("reason=no_cached_chunks", output)
        self.assertNotIn("stage=hybrid.query_embedding", output)
        self.assertNotIn("stage=hybrid.rank_fusion", output)


if __name__ == "__main__":
    unittest.main()
