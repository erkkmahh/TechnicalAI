"""Local Streamlit interface for search, RAG, and verified PDF parameters."""

from inspect import signature
import re

import streamlit as st

import rag
import semantic_search
import hybrid_search
import hybrid_extraction_demo as parameter_workflow
from main import find_manual_matches


def main():
    st.set_page_config(page_title="Technical AI Assistant", layout="centered")
    st.title("Technical AI Assistant")
    st.caption("Search local manuals. Read the original evidence before the Turkish explanation.")

    manuals = sorted(path for path in rag.MANUALS_FOLDER.glob("*")
                     if path.is_file() and path.suffix.lower() == ".pdf")
    with st.sidebar:
        st.header("Local PDF manuals")
        for manual in manuals:
            st.text(manual.name)
        if not manuals:
            st.info("Add PDF manuals to the manuals folder, then refresh this page.")

    document_tab, ai_tab, semantic_tab, hybrid_tab = st.tabs(
        ["Document Search", "AI Assistant", "Semantic Search", "Hybrid Search"])
    with document_tab:
        document_search(manuals)
    with ai_tab:
        ai_assistant(manuals)
    with semantic_tab:
        semantic_search_tab(manuals)
    with hybrid_tab:
        hybrid_search_tab(manuals)


def hybrid_search_tab(manuals):
    st.caption("Experimental: local semantic similarity + Turkish–English FANUC keyword matching.")
    st.caption("Uses existing embeddings only. Build missing caches in Semantic Search first.")
    with st.form("hybrid_search_form"):
        query = st.text_input("Hybrid search query", key="hybrid_query",
                              placeholder="Başka kanaldaki iş miliyle kılavuz çekme")
        submitted = st.form_submit_button("Search hybrid", disabled=not manuals)
    if not submitted:
        return
    if not query.strip():
        st.warning("Please enter a hybrid search query.")
        return
    try:
        with st.spinner("Combining cached semantic and keyword rankings..."):
            results = hybrid_search.search(query, manuals_folder=rag.MANUALS_FOLDER, on_error=st.warning)
        if not results:
            st.info("No indexed text available. Build the local cache using Semantic Search first.")
            return
        st.caption("Keyword terms: " + ", ".join(results[0]["terms"]))
        st.caption("RRF = 1/(60 + semantic rank) + 1/(60 + keyword rank), when a keyword match exists.")
        for result in results:
            st.text(f"{result['filename']} | PDF page: {result['page']} | RRF: {result['rrf_score']:.6f}")
            st.caption(f"Semantic rank: {result['semantic_rank']} | Cosine: {result['semantic_score']:.4f} | "
                       f"Keyword rank: {result['keyword_rank']} | Matched terms: {result['keyword_matches']}")
            st.text(result["text"])
            source = result.get("parameter_source")
            requested = re.fullmatch(
                r"\s*([0-9]{4,5})\s*#\s*([0-7])\s+([A-Z][A-Z0-9_]{1,15})\s*", query)
            if (isinstance(source, dict) and source.get("status") == "VERIFIED"
                    and requested and source.get("filename") == result["filename"]
                    and (source.get("parameter_number"), source.get("bit_number"), source.get("symbol"))
                    == (int(requested[1]), int(requested[2]), requested[3])
                    and source.get("heading_page") == result["page"]
                    and isinstance(source.get("description_pages"), list)
                    and source["description_pages"]
                    and all(type(page) is int and page > 0 for page in source["description_pages"])):
                pages = ", ".join(str(page) for page in source["description_pages"])
                st.caption(
                    f"Doğrulanmış parametre kaynağı | Parametre: {source['parameter_number']} | "
                    f"Bit: {source['bit_number']} | Sembol: {source['symbol']} | "
                    f"Başlık PDF sayfası: {source['heading_page']} | Açıklama PDF sayfası: {pages}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Local hybrid search could not complete: {exc}")


def semantic_search_tab(manuals):
    st.caption("Experimental Turkish-to-English similarity search with local bge-m3 embeddings.")
    st.caption("The first search builds a local cache and can take time. Similarity is not proof of technical correctness.")
    with st.form("semantic_search_form"):
        query = st.text_input("Semantic search query", key="semantic_query",
                              placeholder="Başka bir path üzerindeki spindle ile rigid tapping")
        submitted = st.form_submit_button("Find similar text", disabled=not manuals)
    if not submitted:
        return
    if not query.strip():
        st.warning("Please enter a semantic search query.")
        return
    try:
        with st.spinner("Searching with local embeddings (building cache if needed)..."):
            results = semantic_search.search(query, manuals_folder=rag.MANUALS_FOLDER, on_error=st.warning)
        for result in results:
            st.text(f"{result['filename']} | PDF page: {result['page']} | Similarity: {result['score']:.4f}")
            st.text(result["text"])
        if not results:
            st.info("No readable PDF text found for semantic search.")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Local semantic search could not complete: {exc}")


def document_search(manuals):
    st.caption("Search all local manuals without AI. Keywords and alarm codes are case-insensitive.")
    with st.form("document_search_form"):
        query = st.text_input("Parameter number, alarm code or keyword", key="document_query",
                              placeholder="3703, PS0224, or rigid tapping")
        submitted = st.form_submit_button("Search", disabled=not manuals)
    if not submitted:
        return
    if not query.strip():
        st.warning("Please enter a parameter number, alarm code or keyword.")
        return
    try:
        count = 0
        with st.spinner("Searching all local PDFs..."):
            for match in find_manual_matches(query, rag.MANUALS_FOLDER, on_error=st.warning):
                count += 1
                st.text(f"{match['filename']} | PDF page: {match['page']}")
                st.text(match["text"])
        if count:
            st.caption(f"Matching PDF pages: {count}")
        else:
            st.info("No matching text found in readable PDFs.")
    except OSError as exc:
        st.error(f"Could not search local manuals: {exc}")


def ai_assistant(manuals):
    with st.form("question_form"):
        question = st.text_input("Technical question", key="ai_question", placeholder="What does FANUC parameter 3703#4 SPR do?")
        submitted = st.form_submit_button("Ask", disabled=not manuals)
    if not submitted:
        return
    with rag.performance_timer("total_request"):
        if not question.strip():
            st.warning("Please enter a question containing a parameter number.")
            return

        if show_verified_parameter(question.strip()):
            return

        try:
            with st.spinner("Searching local manuals..."):
                sources = rag.retrieve_context(question.strip())
        except (ValueError, OSError) as exc:
            st.error(str(exc))
            return
        if not sources:
            st.info(rag.NOT_FOUND)
            return

        st.subheader("Original PDF excerpts (English)")
        for source in sources:
            st.text(f"{source['filename']} | PDF page: {source['page']}")
            # Plain text preserves extracted wording and prevents Markdown formatting.
            st.text(source["text"])

        st.subheader("Turkish AI explanation")
        result = {}
        diagnostics = {}
        try:
            with st.spinner("Generating and verifying with local Ollama..."):
                result = rag.ask_ollama(question.strip(), sources)
                evidence = rag.validated_evidence(result, sources)
                # Older loaded backends accept only question, result and sources.
                kwargs = {"diagnostics": diagnostics} if "diagnostics" in signature(rag.verify_answer).parameters else {}
                verified = bool(evidence) and rag.verify_answer(
                    question.strip(), result, sources, **kwargs)
                if not evidence:
                    diagnostics["reason"] = result.get("extraction_error") or "Source evidence validation failed; detailed reason unavailable."
                elif not verified and not diagnostics.get("reason"):
                    diagnostics["reason"] = (rag.meaning_checks(result["answer"], sources)
                                             or "Validation failure reason unavailable.")
            if verified:
                st.text(result["answer"])
                st.caption("Passed the existing source and meaning checks.")
            else:
                st.warning(rag.UNCERTAIN)
                show_ai_debug(result, diagnostics)
        except (ValueError, KeyError, TypeError, AttributeError, OSError) as exc:
            st.warning(rag.UNCERTAIN)
            st.error(f"Local Ollama request could not complete: {exc}")
            if isinstance(result, dict) and "answer" in result:
                diagnostics["reason"] = str(exc)
                show_ai_debug(result, diagnostics)


def show_verified_parameter(question):
    """Handle explicit number#bit requests; never display an unverified draft."""
    if not re.search(r"\d\s*#", question):
        return False
    references = list(re.finditer(r"(?<![\w#])([0-9]{4,5})\s*#\s*([0-7])(?![\w#])", question))
    warning = "NOT_VERIFIED: The parameter could not be verified against the local PDF. No parameter values are shown."
    if len(references) != 1 or question.count("#") != 1:
        st.warning("NOT_VERIFIED: Enter one explicit parameter and bit, for example 10485#1 SUP.")
        return True
    reference = references[0]
    number, bit = map(int, reference.groups())
    symbol_match = re.match(r"[ \t]+([A-Z][A-Z0-9_]{1,15})\b", question[reference.end():])
    try:
        with st.spinner("Extracting and verifying the parameter from the local PDF..."):
            if symbol_match:
                symbol = symbol_match[1]
            else:
                # Resolve an omitted symbol from the PDF, never from the model.
                try:
                    import pymupdf
                except ImportError:
                    import fitz as pymupdf
                with pymupdf.open(str(parameter_workflow.PDF_PATH)) as document:
                    symbols = [match["symbol"] for page in document
                               for match in parameter_workflow.HEADING.finditer(page.get_text("text", sort=False))
                               if parameter_workflow.identity(match)[:2] == (number, bit)]
                if len(symbols) != 1:
                    raise ValueError("The PDF heading is not unique")
                symbol = symbols[0]
            target = (number, bit, symbol)
            metadata, source, isolated = parameter_workflow.extract_source(target, include_layout=True)
            result = parameter_workflow.ask_ollama(source, 180, target=target)
            verification = parameter_workflow.verify(
                result, isolated, target, source_layout=metadata["source_layout"])
        if verification["status"] != "VERIFIED":
            st.warning(warning)
            return True
        main_description = parameter_workflow.extract_main_description(
            isolated, metadata["source_layout"], target)
        st.subheader("Parameter details")
        if main_description["main_description_status"] == "VERIFIED":
            st.caption("Main description: VERIFIED against the local PDF.")
            st.text(main_description["main_description"])
        else:
            st.warning("Main description: NOT_VERIFIED. The description is not shown because its PDF boundaries could not be verified.")
        st.json({field: result[field] for field in parameter_workflow.FIELDS})
        st.text(f"{parameter_workflow.PDF_PATH.name} | PDF page: {metadata['target_page']}")
        st.caption("Five parameter fields: VERIFIED against the local PDF; description whitespace is normalized for comparison.")
    except (ImportError, ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError) as exc:
        st.warning(warning)
        st.error(f"Local parameter extraction could not complete ({type(exc).__name__}).")
    return True


def show_ai_debug(result, diagnostics):
    with st.expander("AI Debug Details", expanded=False):
        st.caption("Unverified Qwen draft — debugging only; do not treat it as a verified explanation.")
        st.code(result.get("raw_answer", result.get("answer", "No raw answer available.")), language=None)
        st.text(diagnostics.get("reason") or "Validation failure reason unavailable.")


if __name__ == "__main__":
    main()
