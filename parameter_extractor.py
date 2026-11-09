"""Conservative, offline extraction of one FANUC parameter-bit text block.

Accepted headings contain four decimal digits, '#', a bit from 0 through 7,
and an uppercase symbol, on one line or two consecutive lines. Horizontal
whitespace is optional around '#', and required before an inline symbol.
Settings must be single-line,
nonempty '0: meaning' and '1: meaning' rows, in that order, separated only
by blank lines. Descriptive text may precede the settings. Multiple headings,
duplicate settings, and disconnected settings are rejected. Prose after the
1 row is not included in its meaning; wrapped meanings cannot be distinguished
from following prose and must be supplied in single-line form.

This is syntax extraction, not semantic verification: meanings are copied
verbatim, never inferred or corrected. Supply one contiguous PDF excerpt;
page/source association is the caller's responsibility. No external services
or third-party dependencies are used.
"""

from dataclasses import dataclass
import re


_HEADING = re.compile(
    r"[ \t]*(?P<parameter>[1-9][0-9]{3})[ \t]*#[ \t]*"
    r"(?P<bit>[0-7])[ \t]+(?P<symbol>[A-Z][A-Z0-9_]*)[ \t]*"
)
_IDENTIFIER = re.compile(
    r"[ \t]*(?P<parameter>[1-9][0-9]{3})[ \t]*#[ \t]*(?P<bit>[0-7])[ \t]*"
)
_SYMBOL = re.compile(r"[ \t]*([A-Z][A-Z0-9_]*)[ \t]*")
# A parameter without a bit also starts a new block. Fail closed rather than
# borrowing its rows for the preceding bit. This includes split table headings.
_PARAMETER_BOUNDARY = re.compile(r"[ \t]*[1-9][0-9]{3}(?:[ \t]+.*)?[ \t]*")
_SETTING = re.compile(r"[ \t]*(?P<value>[01])[ \t]*:[ \t]*(?P<meaning>\S(?:.*\S)?)[ \t]*")
_SETTING_LIKE = re.compile(r"[ \t]*[0-9]+[ \t]*:")


@dataclass(frozen=True)
class Setting:
    value: int
    meaning: str
    text: str


@dataclass(frozen=True)
class ParameterBit:
    parameter_number: int
    bit_number: int
    symbol: str
    heading: str
    settings: tuple[Setting, Setting]

    @property
    def identifier(self) -> str:
        return f"{self.parameter_number}#{self.bit_number}"


def extract_parameter_bit(text: str) -> ParameterBit | None:
    """Return one unambiguous complete record, or None for unsupported text.

    CRLF/CR line endings are accepted; original line contents (including
    indentation and trailing spaces) are retained in heading and setting.text.
    Supply exactly one parameter block. Multiple parameter headings (including
    headings without bits) are rejected, even if only one block is complete.
    Prose after the pair is allowed but never appended to either setting.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    headings = []
    for index, line in enumerate(lines):
        if match := _HEADING.fullmatch(line):
            headings.append((index, index, match, match['symbol']))
        elif (match := _IDENTIFIER.fullmatch(line)) and index + 1 < len(lines):
            if symbol := _SYMBOL.fullmatch(lines[index + 1]):
                headings.append((index, index + 1, match, symbol[1]))
    # A stray/malformed bit heading must not be silently ignored.
    if len(headings) != 1 or sum("#" in line for line in lines) != 1:
        return None
    heading_index, heading_end, heading, symbol = headings[0]
    if any(_PARAMETER_BOUNDARY.fullmatch(line) for index, line in enumerate(lines)
           if index != heading_index):
        return None
    rows = []
    row_indices = []
    for index, line in enumerate(lines):
        if not _SETTING_LIKE.match(line):
            continue
        match = _SETTING.fullmatch(line)
        if match is None or index <= heading_end:
            return None
        rows.append(Setting(int(match['value']), match['meaning'], line))
        row_indices.append(index)
    if len(rows) != 2 or [row.value for row in rows] != [0, 1]:
        return None
    if any(line.strip() for line in lines[row_indices[0] + 1:row_indices[1]]):
        return None
    return ParameterBit(
        int(heading['parameter']), int(heading['bit']), symbol,
        "\n".join(lines[heading_index:heading_end + 1]), (rows[0], rows[1]),
    )
