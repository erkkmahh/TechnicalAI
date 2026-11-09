"""Inspect local PDF text around 3703 or SPR without using external services.

Usage: python inspect_parameter_layout.py "C:\\path\\to\\manual.pdf"
Page numbers are one-based. Each matching line gets at most 15 lines of
context (including itself). Overlapping excerpts are intentionally retained.
Text is printed in PyMuPDF's extraction order without stripping or reflowing.
Use --page N to print a whole page with numbered lines; --lines N limits
output to its first N lines, or --lines START-END selects an inclusive range.
"""

import argparse
from pathlib import Path
import re
import sys


MATCH = re.compile(r"(?<!\w)(?:3703|SPR)(?!\w)", re.IGNORECASE)


def positive_integer(value):
    try:
        number = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("expected a positive integer") from None
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def line_range(value):
    match = re.fullmatch(r"([0-9]+)(?:[-:]([0-9]+))?", value)
    if match is None:
        raise argparse.ArgumentTypeError("expected N or START-END (one-based)")
    first = int(match[1]) if match[2] else 1
    last = int(match[2] or match[1])
    if first < 1 or last < first:
        raise argparse.ArgumentTypeError("line range must be positive and ascending")
    return first, last


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a local PDF file")
    parser.add_argument("--page", type=positive_integer, help="Print this one-based PDF page with numbered lines")
    parser.add_argument("--lines", type=line_range, metavar="N|START-END",
                        help="With --page, print the first N lines or an inclusive line range")
    args = parser.parse_args()
    if args.lines is not None and args.page is None:
        parser.error("--lines requires --page")

    try:
        import pymupdf
    except ImportError:
        try:
            import fitz as pymupdf
        except ImportError:
            parser.exit(1, "PyMuPDF is required. Install it with: python -m pip install PyMuPDF\n")

    if not args.pdf.is_file():
        parser.exit(1, f"PDF file not found: {args.pdf}\n")

    # Avoid newline translation so extracted line endings remain unchanged.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(newline="")

    found = False
    try:
        with pymupdf.open(str(args.pdf)) as document:
            if not document.is_pdf:
                parser.exit(1, "The supplied file is not a PDF.\n")
            if document.needs_pass:
                parser.exit(1, "The PDF requires a password; provide an unlocked local copy.\n")
            if args.page is not None:
                if args.page > len(document):
                    parser.error(f"--page must be between 1 and {len(document)}")
                lines = document[args.page - 1].get_text("text", sort=False).splitlines(keepends=True)
                first, last = args.lines or (1, len(lines))
                print(f"--- Page {args.page} ({len(lines)} extracted lines) ---")
                for number in range(first, min(last, len(lines)) + 1):
                    line = lines[number - 1]
                    sys.stdout.write(f"{number}: {line}")
                    if not line.endswith(("\n", "\r")):
                        sys.stdout.write("\n")
                return
            for page_number, page in enumerate(document, start=1):
                lines = page.get_text("text", sort=False).splitlines(keepends=True)
                for index, line in enumerate(lines):
                    if MATCH.search(line) is None:
                        continue
                    found = True
                    start = max(0, min(index - 7, len(lines) - 15))
                    end = min(len(lines), start + 15)
                    print(
                        f"--- Page {page_number}, matching line {index + 1}, "
                        f"extracted lines {start + 1}-{end} ---"
                    )
                    excerpt = "".join(lines[start:end])
                    sys.stdout.write(excerpt)
                    # Separate metadata from an excerpt with no final newline.
                    if not excerpt.endswith(("\n", "\r")):
                        sys.stdout.write("\n")
                    print("--- End excerpt ---")
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Could not inspect PDF: {exc}\n")

    if not found:
        print("No occurrences of 3703 or SPR found in extracted text. Scanned pages may require OCR.")


if __name__ == "__main__":
    main()
