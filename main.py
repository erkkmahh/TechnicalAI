"""Search local PDF manuals for a parameter number."""

from pathlib import Path
import re
import sys

import pymupdf


# Find manuals beside this file, even when run from another folder.
MANUALS_FOLDER = Path(__file__).resolve().parent / "manuals"


def find_manual_matches(query, manuals_folder=None, on_error=print):
    """Yield matching page excerpts for numbers, alarm codes, or literal keywords."""
    query = query.strip()
    if not query:
        return
    folder = MANUALS_FOLDER if manuals_folder is None else manuals_folder
    pdf_files = sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    # Match 3703, P3703, or 3703.0, but not 13703 or 37030.
    expression = re.escape(query)
    if query.isdigit():
        expression = r"(?<!\d)" + expression + r"(?!\d)"
    pattern = re.compile(expression, re.IGNORECASE)

    for pdf_path in pdf_files:
        try:
            with pymupdf.open(pdf_path) as document:
                if document.needs_pass:
                    on_error(f"Skipping password-protected PDF: {pdf_path.name}")
                    continue

                for page_index, page in enumerate(document):
                    lines = page.get_text("text", sort=True).splitlines()
                    blocks = []
                    for line_index, line in enumerate(lines):
                        if pattern.search(line):
                            # Slice ends are exclusive: +11 includes 10 lines after.
                            start = max(0, line_index - 5)
                            end = min(len(lines), line_index + 11)
                            # Merge overlapping or touching blocks to avoid repeats.
                            if blocks and start <= blocks[-1][1]:
                                blocks[-1][1] = max(blocks[-1][1], end)
                            else:
                                blocks.append([start, end])

                    if blocks:
                        text = "\n...\n".join("\n".join(lines[start:end]) for start, end in blocks)
                        yield {"filename": pdf_path.name, "page": page_index + 1, "text": text}
        except (pymupdf.FileDataError, OSError, RuntimeError) as error:
            on_error(f"Could not read {pdf_path.name}: {error}")


def search_manuals(parameter):
    """Print matching pages using the same search as the document tab."""
    if not any(path.is_file() and path.suffix.lower() == ".pdf"
               for path in MANUALS_FOLDER.iterdir()):
        print("No PDF files found. Add a PDF to the manuals folder first.")
        return
    matching_pages = 0
    for match in find_manual_matches(parameter):
        matching_pages += 1
        print(f"\nFile: {match['filename']} | PDF page: {match['page']}")
        for line in match["text"].splitlines():
            print(f"  {line.strip()}")

    if matching_pages:
        print(f"\nFound matches on {matching_pages} PDF page(s).")
    else:
        print(f"\nNo matches found for parameter {parameter} in readable PDFs.")


def main():
    """Choose the original search or a local question-answer session."""
    # Windows terminals/pipes may otherwise reject Turkish characters.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Technical AI Assistant — local PDF search")
    mode = input("1: Parameter search | 2: Ask local Ollama [1]: ").strip()
    if mode == "2":
        from rag import answer_question

        answer_question(input("Your question: ").strip())
        return
    if mode not in ("", "1"):
        print("Please choose 1 or 2.")
        return
    parameter = input("Enter a parameter number (for example, 3703): ").strip()
    if not parameter or any(character not in "0123456789" for character in parameter):
        print("Please enter digits only, for example: 3703.")
        return

    search_manuals(parameter)


if __name__ == "__main__":
    main()
