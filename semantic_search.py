"""Experimental local multilingual retrieval, independent of the RAG pipeline."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from urllib import error, request

import pymupdf

from main import MANUALS_FOLDER


MODEL = "bge-m3"
EMBED_URL = "http://127.0.0.1:11434/api/embed"
CACHE_FOLDER = Path(__file__).resolve().parent / ".semantic_cache"
CHUNK_SIZE = 1200  # Characters, not tokens.
OVERLAP = 200
BATCH_SIZE = 16
CACHE_VERSION = 1


def extract_chunks(pdf_path, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """Keep overlapping character windows within each PDF page."""
    if not 0 <= overlap < chunk_size:
        raise ValueError("Overlap must be smaller than the positive chunk size.")
    chunks = []
    with pymupdf.open(pdf_path) as pdf:
        if pdf.needs_pass:
            raise pymupdf.FileDataError(f"Password-protected PDF: {pdf_path.name}")
        for page_index, page in enumerate(pdf):
            text = page.get_text("text", sort=True)
            for start in range(0, len(text), chunk_size - overlap):
                excerpt = text[start:start + chunk_size]
                if excerpt.strip():
                    chunks.append({"filename": pdf_path.name, "page": page_index + 1,
                                   "text": excerpt})
                if start + chunk_size >= len(text):
                    break
    return chunks


def normalized(vector):
    """Validate a vector and scale its length to one."""
    if not isinstance(vector, list) or not vector:
        raise ValueError("Invalid embedding vector.")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in vector):
        raise ValueError("Embedding contains non-finite or non-numeric values.")
    length = math.hypot(*vector)
    if not length or not math.isfinite(length):
        raise ValueError("Embedding has invalid length.")
    return [value / length for value in vector]


def embed_texts(texts):
    """Call only the installed local model. Never pull or download a model."""
    if not texts:
        return []
    payload = {"model": MODEL, "input": texts, "truncate": False}
    req = request.Request(EMBED_URL, data=json.dumps(payload).encode("utf-8"),
                          headers={"Content-Type": "application/json"})

    class NoRedirect(request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            return None

    opener = request.build_opener(request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(req, timeout=180) as response:
            vectors = json.load(response)["embeddings"]
    except error.HTTPError as exc:
        raise ValueError(f"Local Ollama embedding error {exc.code}. Check that {MODEL} is installed.") from exc
    if not isinstance(vectors, list) or len(vectors) != len(texts):
        raise ValueError("Ollama returned the wrong number of embeddings.")
    vectors = [normalized(vector) for vector in vectors]
    if any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("Ollama returned inconsistent embedding dimensions.")
    return vectors


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as pdf_file:
        for block in iter(lambda: pdf_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def document_embeddings(pdf_path, cache_folder):
    """Reuse a PDF's cache unless its contents or embedding settings changed."""
    settings = [str(pdf_path.resolve()), MODEL, CHUNK_SIZE, OVERLAP, CACHE_VERSION]
    key = hashlib.sha256(json.dumps(settings).encode("utf-8")).hexdigest()
    cache_path = cache_folder / f"{key}.json"
    fingerprint = file_hash(pdf_path)
    try:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["fingerprint"] == fingerprint:
            chunks = cached["chunks"]
            vectors = [normalized(vector) for vector in cached["embeddings"]]
            if not isinstance(chunks, list) or len(chunks) != len(vectors):
                raise ValueError("Invalid cache length.")
            for chunk in chunks:
                if (chunk["filename"] != pdf_path.name or type(chunk["page"]) is not int
                        or chunk["page"] < 1 or not isinstance(chunk["text"], str)):
                    raise ValueError("Invalid cached source.")
            return chunks, vectors
    except (OSError, ValueError, KeyError, TypeError):
        pass  # Missing or damaged cache: rebuild this PDF only.

    chunks = extract_chunks(pdf_path)
    vectors = []
    for start in range(0, len(chunks), BATCH_SIZE):
        vectors.extend(embed_texts([chunk["text"] for chunk in chunks[start:start + BATCH_SIZE]]))
    cache_folder.mkdir(parents=True, exist_ok=True)
    # Replace only complete files, so a failed build cannot leave a partial cache.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_folder,
                                         suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            json.dump({"fingerprint": fingerprint, "chunks": chunks, "embeddings": vectors}, temporary)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return chunks, vectors


def cosine_similarity(left, right):
    if len(left) != len(right):
        raise ValueError("Embedding dimensions differ. Clear .semantic_cache after changing the model.")
    return max(-1.0, min(1.0, sum(a * b for a, b in zip(normalized(left), normalized(right)))))


def search(query, manuals_folder=MANUALS_FOLDER, cache_folder=CACHE_FOLDER, on_error=print):
    """Return up to five ranked excerpts, without generating an explanation."""
    if not query.strip():
        raise ValueError("Please enter a semantic search query.")
    documents = []
    for pdf_path in sorted(manuals_folder.glob("*")):
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            continue
        try:
            chunks, vectors = document_embeddings(pdf_path, cache_folder)
        except (pymupdf.FileDataError, RuntimeError) as exc:
            on_error(f"Could not read {pdf_path.name}: {exc}")
            continue
        documents.extend(zip(chunks, vectors))
    if not documents:
        return []
    query_vector = embed_texts([query.strip()])[0]
    results = [{**chunk, "score": cosine_similarity(query_vector, vector)}
               for chunk, vector in documents]
    return sorted(results, key=lambda result: result["score"], reverse=True)[:5]
