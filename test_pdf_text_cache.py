"""Offline PDF text cache regressions using short synthetic manuals."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf
import rag


class PdfTextCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.path = self.folder / "manual.pdf"
        rag._PAGE_TEXT_CACHE.clear()
        self.addCleanup(rag._PAGE_TEXT_CACHE.clear)

    def write_pdf(self, meaning="Not performed"):
        with pymupdf.open() as pdf:
            for text in ("3703 #3 MPP", f"Selection is:\n0: {meaning}\n1: Performed\n3703 #4 SPR\nUnrelated"):
                pdf.new_page().insert_text((50, 50), text)
            pdf.save(self.path)

    def test_reuse_preserves_sources_and_cross_page_evidence(self):
        self.write_pdf()
        with patch("rag.pymupdf.open", wraps=pymupdf.open) as opened:
            first = rag.retrieve_context("Explain 3703#3 MPP", self.folder)
            with self.assertLogs(rag._performance_log) as logs:
                second = rag.retrieve_context("Explain 3703#3 MPP", self.folder)
            self.assertEqual(opened.call_count, 1)
        self.assertEqual(first, second)
        self.assertEqual([s["page"] for s in second], [1, 2])
        evidence = rag.extract_parameter_evidence("Explain 3703#3 MPP", second)
        self.assertEqual([r["page"] for r in evidence["settings"]], [2, 2])
        output = "\n".join(logs.output)
        self.assertIn("status=hit", output)
        self.assertNotIn("stage=retrieval.pdf_read", output)
        self.assertNotIn("stage=retrieval.pdf_open", output)

    def test_changed_pdf_invalidates_text(self):
        self.write_pdf()
        before = self.path.stat()
        with patch("rag.pymupdf.open", wraps=pymupdf.open) as opened:
            first = rag.retrieve_context("Explain 3703#3 MPP", self.folder)
            self.write_pdf("Disabled")
            os.utime(self.path, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
            opened.reset_mock()  # Exclude creating the replacement synthetic PDF.
            second = rag.retrieve_context("Explain 3703#3 MPP", self.folder)
            self.assertEqual(opened.call_count, 1)
        self.assertIn("0: Not performed", first[1]["text"])
        self.assertIn("0: Disabled", second[1]["text"])
        self.assertNotIn("Not performed", second[1]["text"])

    def test_failed_refresh_does_not_return_stale_text(self):
        self.write_pdf()
        rag.retrieve_context("Explain 3703#3 MPP", self.folder)
        self.path.write_bytes(b"not a PDF")
        self.assertEqual(rag.retrieve_context("Explain 3703#3 MPP", self.folder), [])
        self.assertNotIn(self.path.resolve(), rag._PAGE_TEXT_CACHE)

    def test_modified_during_read_is_not_cached(self):
        self.write_pdf()
        signature = rag._pdf_signature(self.path)
        with patch("rag._pdf_signature", side_effect=[signature, (*signature[:-1], signature[-1] + 1)]):
            with self.assertRaisesRegex(OSError, "changed during text extraction"):
                rag._cached_pdf_lines(self.path)
        self.assertNotIn(self.path.resolve(), rag._PAGE_TEXT_CACHE)


if __name__ == "__main__":
    unittest.main()
