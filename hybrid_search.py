"""Experimental rank fusion over the existing local semantic cache only."""

import hashlib
import json
import re
from pathlib import Path

import hybrid_extraction_demo as parameter_parser

import semantic_search as semantic
from rag import performance_timer, _performance_log


TERMINOLOGY = {
    "birinci eksen": "first axis",
    "ikinci eksen": "second axis",
    "takım ekseni": "tool axis",
    "seçim sinyali": "selection signal",
    "eksen yönü": "axis direction",
    "manuel el çarkı": "manual handle",
    "rijit kılavuz çekme": "rigid tapping",
    "kılavuz çekme": "tapping",
    "iş mili": "spindle",
    "kanal": "path",
}
STOPWORDS = {"bir", "ile", "ve", "ne", "nedir", "nasıl", "başka", "için", "hangi",
             "what", "does", "the", "of", "is", "a", "an", "with", "in", "do"}


def normalized_text(text):
    text = " ".join(text.casefold().replace("\u0307", "").split())
    # Manuals use both written ordinals and abbreviations such as '1st axis'.
    text = re.sub(r"\b1st\b", "first", text)
    return re.sub(r"\b2nd\b", "second", text)


def phrase_pattern(term):
    """Match a complete phrase, allowing ordinary spaces or hyphens."""
    return r"[\s-]+".join(re.escape(word) for word in term.split())


def query_terms(query):
    """Map explicit phrases, including a few common Turkish suffixes."""
    remaining = normalized_text(query)
    terms = []
    # Long phrases first: rigid tapping is one concept, not two duplicate votes.
    phrases = sorted(TERMINOLOGY.items(), key=lambda item: (-len(item[0].split()), -len(item[0])))
    for turkish, english in phrases:
        suffix = (r"(?:daki|deki|da|de|dan|den|ın|in|un|ün|nın|nin|nun|nün|ı|i|u|ü|ni|nı|nu|nü"
                  r"|yla|yle|ları|leri|k|ne|na|nde|nda|nden|ndan|ndeki|ndaki|e|a|ler|lar|lerde|larda|lerin|ların)?")
        pattern = r"\b(?:" + phrase_pattern(turkish) + suffix + "|" + phrase_pattern(english) + r"s?)\b"
        if re.search(pattern, remaining):
            terms.append(english)
            remaining = re.sub(pattern, " ", remaining)
    terms.extend(token for token in re.findall(r"\w+", remaining) if token not in STOPWORDS)
    return list(dict.fromkeys(terms))


def keyword_ranking(chunks, terms):
    """Rank by complete phrase coverage first, then total distinct-term coverage."""
    counts = {}
    phrase_counts = {}
    for index, chunk in enumerate(chunks):
        text = normalized_text(chunk["text"])
        matches = [bool(re.search(r"\b" + phrase_pattern(term) + (r"s?" if term[-1].isalpha() else "")
                                  + r"\b", text)) for term in terms]
        count = sum(matches)
        if count:
            counts[index] = count
            phrase_counts[index] = sum(hit for term, hit in zip(terms, matches) if " " in term)
    return sorted(counts, key=lambda index: (-phrase_counts[index], -counts[index], index)), counts


def reciprocal_rank_fusion(semantic_order, keyword_order, k=60):
    """Equal-weight RRF: sum 1/(k + rank), with ranks starting at one."""
    if k <= 0:
        raise ValueError("RRF k must be positive.")
    scores = {}
    for order in (semantic_order, keyword_order):
        # A chunk contributes at most once per ranking.
        for rank, index in enumerate(dict.fromkeys(order), start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def load_cached_chunks(manuals_folder, cache_folder, on_error=print):
    """Read valid existing caches. Never extract PDFs, rebuild, or write caches."""
    chunks, vectors = [], []
    for path in sorted(manuals_folder.glob("*")):
        if not path.is_file() or path.suffix.lower() != ".pdf":
            continue
        settings = [str(path.resolve()), semantic.MODEL, semantic.CHUNK_SIZE,
                    semantic.OVERLAP, semantic.CACHE_VERSION]
        key = hashlib.sha256(json.dumps(settings).encode("utf-8")).hexdigest()
        try:
            with performance_timer("hybrid.cache_read"):
                cached = json.loads((cache_folder / f"{key}.json").read_text(encoding="utf-8"))
            with performance_timer("hybrid.pdf_fingerprint_read"):
                fingerprint = semantic.file_hash(path)
            if cached["fingerprint"] != fingerprint:
                raise ValueError("PDF has changed")
            pdf_chunks = cached["chunks"]
            pdf_vectors = [semantic.normalized(vector) for vector in cached["embeddings"]]
            if not isinstance(pdf_chunks, list) or len(pdf_chunks) != len(pdf_vectors):
                raise ValueError("invalid cache length")
            for chunk in pdf_chunks:
                if (chunk["filename"] != path.name or type(chunk["page"]) is not int
                        or chunk["page"] < 1 or not isinstance(chunk["text"], str)):
                    raise ValueError("invalid cached source")
            chunks.extend(pdf_chunks)
            vectors.extend(pdf_vectors)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            on_error(f"Cache unavailable for {path.name}: {exc}. Use Semantic Search to build/update it first.")
    return chunks, vectors


def expand_parameter_sources(query, results, manuals_folder):
    """Attach verified PDF sections without changing retrieval fields or ordering."""
    match = re.fullmatch(r"\s*([0-9]{4,5})\s*#\s*([0-7])\s+([A-Z][A-Z0-9_]{1,15})\s*", query)
    if not match:
        return results
    target = (int(match[1]), int(match[2]), match[3])
    sections = {}
    for result in results:
        if not any(parameter_parser.identity(h) == target
                   for h in parameter_parser.HEADING.finditer(result["text"])):
            continue
        filename = result["filename"]
        if filename not in sections:
            sections[filename] = None
            try:
                folder = Path(manuals_folder).resolve()
                path = (folder / filename).resolve()
                if path.parent != folder or not path.is_file():
                    continue
                metadata, _, isolated = parameter_parser.extract_source(
                    target, include_layout=True, pdf_path=path)
                layout = metadata.get("source_layout")
                if (metadata.get("mode") != "parameter_range" or metadata.get("fallback_reason")
                        or isolated is None or not layout
                        or [row["text"] for row in layout] != isolated.splitlines()
                        or metadata["target_page"] != result["page"]):
                    continue
                headings = list(parameter_parser.HEADING.finditer(isolated))
                if (len(headings) != 1 or headings[0].start() != 0
                        or parameter_parser.identity(headings[0]) != target):
                    continue
                description = parameter_parser.extract_main_description(isolated, layout, target)
                if description["main_description_status"] != "VERIFIED":
                    continue
                sections[filename] = {
                    "status": "VERIFIED", "filename": filename,
                    "parameter_number": target[0], "bit_number": target[1], "symbol": target[2],
                    "heading_page": metadata["target_page"],
                    "pages": sorted({row["page"] for row in layout}),
                    "description_pages": sorted({layout[i]["page"] for i in description["source_line_indices"]}),
                    "text": isolated, "main_description": description["main_description"],
                }
            except (OSError, ValueError, RuntimeError, KeyError, TypeError, IndexError, AttributeError):
                continue
        section = sections[filename]
        if section is not None and section["heading_page"] == result["page"]:
            result["parameter_source"] = dict(section)
    return results


def search(query, manuals_folder=semantic.MANUALS_FOLDER, cache_folder=semantic.CACHE_FOLDER, on_error=print):
    if not query.strip():
        raise ValueError("Please enter a hybrid search query.")
    _performance_log.info("retrieval path=hybrid unavailable_stages=pdf_text_extraction reason=cached_chunks_only")
    with performance_timer("hybrid.cache_loading"):
        chunks, vectors = load_cached_chunks(manuals_folder, cache_folder, on_error)
    if not chunks:
        _performance_log.info("retrieval path=hybrid unavailable_stages=query_embedding,semantic_scoring,keyword_search,rank_fusion reason=no_cached_chunks")
        return []
    # Only the original query is embedded; dictionary expansion is lexical only.
    with performance_timer("hybrid.query_embedding"):
        query_vector = semantic.embed_texts([query.strip()])[0]
    with performance_timer("hybrid.semantic_scoring"):
        similarities = [semantic.cosine_similarity(query_vector, vector) for vector in vectors]
        semantic_order = sorted(range(len(chunks)), key=lambda index: (-similarities[index], index))
    with performance_timer("hybrid.keyword_search"):
        terms = query_terms(query)
        keyword_order, counts = keyword_ranking(chunks, terms)
    semantic_ranks = {index: rank for rank, index in enumerate(semantic_order, start=1)}
    keyword_ranks = {index: rank for rank, index in enumerate(keyword_order, start=1)}
    with performance_timer("hybrid.rank_fusion"):
        fused = reciprocal_rank_fusion(semantic_order, keyword_order)[:5]
    results = [{**chunks[index], "chunk_id": index, "rrf_score": score,
             "semantic_score": similarities[index], "semantic_rank": semantic_ranks[index],
             "keyword_rank": keyword_ranks.get(index), "keyword_matches": counts.get(index, 0),
             "terms": terms}
            for index, score in fused]

    return expand_parameter_sources(query, results, manuals_folder)
