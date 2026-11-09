"""Small conservative checks, independent of the language model."""

import re


def setting_segments(text):
    """Separate setting sentences without dropping the surrounding explanation."""
    return re.split(r"\n|(?<=[.!?])\s+(?=[01]\s*:)", text)


def setting_rows(text):
    """Collect explicit 0/1 rows, keeping their values attached to their text."""
    # Only these demonstrated translations are equivalent. Other descriptions
    # retain their full text, including qualifiers, for the existing comparison.
    equivalents = {"not performed": "Not performed", "yapılmaz": "Not performed",
                   "performed": "Performed", "yapılır": "Performed"}
    rows = {}
    for segment in setting_segments(text):
        # Allow '0: Yapılmaz, 1: Yapılır' as well as separate lines. Split only
        # immediately before another value label, never inside its explanation.
        parts = re.split(r"(?:[,;]\s*|\s+ve\s+)(?=[01]\s*:)", segment)
        matches = [re.fullmatch(r"\s*(?:[-*]\s*)?([01])\s*:\s*(.+?)\s*", part)
                   for part in parts]
        if all(matches):
            for match in matches:
                value = " ".join(match[2].split())
                value = equivalents.get(value.rstrip(".").casefold(), value)
                rows.setdefault(match[1], set()).add(value)
    return rows


def meaning_checks(answer, sources):
    """Return a rejection reason, or an empty string when basic checks pass."""
    context = "\n".join(source["text"] for source in sources)
    expected = setting_rows(context)
    actual = setting_rows(answer)
    # Compare value meanings; accept only the narrow translations above.
    # Ambiguous rows from multiple parameter descriptions require abstention.
    if any(len(values) != 1 for values in expected.values()):
        return "The retrieved context contains conflicting setting rows."
    if actual != expected:
        return "The answer omitted, changed, added, or reversed a 0/1 setting row."

    if len(expected) == 2 and expected["0"] != expected["1"]:
        # A small model can approve its own mistranslation. Do not trust it to
        # verify negated or unconditional action claims in the free prose.
        # Keep polarity in the verbatim value rows and use neutral prose.
        prose = "\n".join(segment for segment in setting_segments(answer)
                          if not setting_rows(segment)).lower()
        risky = (
            r"\b(?:her\s+zaman|daima|asla|always|never|not|cannot|can't|doesn't)\b"
            r"|\b(?:engell\w*|yasak\w*|önle\w*|iptal\w*|kapat\w*|değil\w*"
            r"|yapılmaz\w*|gerçekleşmez\w*|kullanılmaz\w*|olmaz\w*"
            r"|disable\w*|prevent\w*|block\w*|enable\w*|etkinleştir\w*)\b"
            r"|\bdevre\s+dışı\w*|\bdevreye\s+al\w*"
        )
        if re.search(risky, prose):
            return "Negated or unconditional prose cannot be safely verified against the value-dependent behavior."

    context_lower = " ".join(context.lower().split())
    answer_lower = " ".join(answer.lower().split())
    relations = ("another path", "other path", "same path")
    for phrase in relations:
        in_source = bool(re.search(r"\b" + phrase + r"\b", context_lower))
        in_answer = bool(re.search(r"\b" + phrase + r"\b", answer_lower))
        if in_source != in_answer:
            return "The answer did not preserve the source's path relationship."
    # Catch the reported reversal even if the correct English words appear too.
    if ("another path" in context_lower or "other path" in context_lower) and "same path" not in context_lower:
        if re.search(r"\baynı\s+(?:path\b|yol\w*)", answer_lower):
            return "The answer changed another path to the same path."
    return ""
