"""Offline semantic retrieval tests. No requests reach an Ollama server."""

import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pymupdf

import semantic_search as semantic


class SemanticTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.manuals = self.folder / "manuals"
        self.manuals.mkdir()
        self.cache = self.folder / "cache"

    def pdf(self, name, pages):
        path = self.manuals / name
        with pymupdf.open() as pdf:
            for text in pages:
                page = pdf.new_page()
                page.insert_text((50, 50), text, fontsize=10)
            pdf.save(path)
        return path

    def fake_embeddings(self, texts):
        return [[1.0, 0.0] if ("spindle" in text or "mil" in text) else [0.0, 1.0]
                for text in texts]

    def test_chunks_overlap_and_preserve_pages(self):
        path = self.pdf("sample.pdf", ["\n".join(f"Line {i}: spindle control explanation." for i in range(20)),
                                        "Second page text."])
        chunks = semantic.extract_chunks(path, chunk_size=100, overlap=20)
        first_page = [chunk for chunk in chunks if chunk["page"] == 1]
        self.assertGreater(len(first_page), 1)
        for left, right in zip(first_page, first_page[1:]):
            self.assertEqual(left["text"][-20:], right["text"][:20])
        with pymupdf.open(path) as pdf:
            reconstructed = first_page[0]["text"] + "".join(c["text"][20:] for c in first_page[1:])
            self.assertEqual(reconstructed, pdf[0].get_text("text", sort=True))
        self.assertEqual(chunks[-1]["page"], 2)
        self.assertTrue(all(c["filename"] == "sample.pdf" for c in chunks))

    def test_top_five_cosine_results(self):
        self.pdf("sample.pdf", ["spindle selection"] + [f"Unrelated alarm {i}" for i in range(6)])
        with patch("semantic_search.embed_texts", side_effect=self.fake_embeddings):
            results = semantic.search("mil seçimi", self.manuals, self.cache)
        self.assertEqual(len(results), 5)
        self.assertEqual(results[0]["page"], 1)
        self.assertEqual(results[0]["score"], 1.0)
        self.assertEqual([r["score"] for r in results], sorted([r["score"] for r in results], reverse=True))
        self.assertAlmostEqual(semantic.cosine_similarity([3, 4], [6, 8]), 1.0)
        self.assertAlmostEqual(semantic.cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_cache_skips_pdf_extraction_and_document_embedding(self):
        self.pdf("sample.pdf", ["spindle selection"])
        with patch("semantic_search.embed_texts", side_effect=self.fake_embeddings):
            semantic.search("mil", self.manuals, self.cache)
        with patch("semantic_search.extract_chunks", side_effect=AssertionError("PDF reprocessed")), \
             patch("semantic_search.embed_texts", side_effect=self.fake_embeddings) as embed:
            results = semantic.search("mil", self.manuals, self.cache)
            embed.assert_called_once_with(["mil"])
        self.assertEqual(results[0]["filename"], "sample.pdf")

    def test_changed_pdf_and_settings_rebuild_cache(self):
        path = self.pdf("sample.pdf", ["spindle selection"])
        with patch("semantic_search.embed_texts", side_effect=self.fake_embeddings):
            semantic.search("mil", self.manuals, self.cache)
            # Replace test data only; no user manuals are involved.
            path.unlink()
            self.pdf("sample.pdf", ["New spindle selection"])
            with patch("semantic_search.extract_chunks", wraps=semantic.extract_chunks) as extract:
                semantic.search("mil", self.manuals, self.cache)
                extract.assert_called_once()
            with patch.object(semantic, "MODEL", "another-local-model"), \
                 patch("semantic_search.extract_chunks", wraps=semantic.extract_chunks) as extract:
                semantic.search("mil", self.manuals, self.cache)
                extract.assert_called_once()

    def test_corrupt_cache_rebuilds_and_deleted_pdf_is_not_returned(self):
        path = self.pdf("sample.pdf", ["spindle selection"])
        with patch("semantic_search.embed_texts", side_effect=self.fake_embeddings):
            semantic.search("mil", self.manuals, self.cache)
            next(self.cache.glob("*.json")).write_text("broken", encoding="utf-8")
            self.assertTrue(semantic.search("mil", self.manuals, self.cache))
        path.unlink()
        with patch("semantic_search.embed_texts") as embed:
            self.assertEqual(semantic.search("mil", self.manuals, self.cache), [])
            embed.assert_not_called()

    def test_local_transport_and_bad_responses(self):
        from io import StringIO

        with patch("semantic_search.request.build_opener") as build:
            build.return_value.open.return_value.__enter__.return_value = StringIO('{"embeddings": [[3, 4]]}')
            self.assertEqual(semantic.embed_texts(["test"]), [[0.6, 0.8]])
            req = build.return_value.open.call_args.args[0]
            self.assertEqual(req.full_url, "http://127.0.0.1:11434/api/embed")
            self.assertEqual(json.loads(req.data)["model"], "bge-m3")
            self.assertFalse(json.loads(req.data)["truncate"])
            self.assertEqual(build.call_args.args[0].proxies, {})
            self.assertIsNone(build.call_args.args[1].redirect_request(None, None, 302, "", {}, "https://example.com"))
        for vector in [[], [0, 0], [math.nan, 1], ["x", 1]]:
            with self.assertRaises(ValueError):
                semantic.normalized(vector)
        with self.assertRaises(ValueError):
            semantic.cosine_similarity([1, 0], [1])
        with patch("semantic_search.embed_texts") as embed:
            with self.assertRaises(ValueError):
                semantic.search("  ", self.manuals, self.cache)
            embed.assert_not_called()


if __name__ == "__main__":
    unittest.main()
