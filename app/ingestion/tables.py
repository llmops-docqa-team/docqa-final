"""Table extraction (PyMuPDF find_tables) and Markdown rendering.

Each table becomes its own chunk. A long table is split by rows and the header row is repeated in every
piece, so every piece is self-describing for retrieval and for the LLM.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.ingestion.tokens import estimate_tokens
from app.observability.logging import get_logger

log = get_logger("ingestion.tables")

BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class Table:
    bbox: BBox
    rows: list[list[str]]  # rows[0] is the header
    title: str = ""  # nearby text above the table ("Standalone Balance Sheet ... (All amount in ...)")


# Many Indian annual reports draw the rupee sign with a font that maps it to a backtick, so PyMuPDF
# extracts "₹ 3,124.83" as "` 3,124.83" (and "in ₹ million" as "in ` million"). A backtick before an
# amount or a unit word is restored to "₹", which also keeps it from opening a Markdown code span in the UI.
_RUPEE_GLYPH = re.compile(
    r"`(?=\s?(?:[\d(]|million|crore|lakh|billion|thousand)\b|\sin\b)"  # "` 3,124", "` million", "(` in"
    r"|(?<=\bin\s)`(?=\))",  # "Basic (in `)"
    re.IGNORECASE,
)


def fix_rupee(text: str) -> str:
    return _RUPEE_GLYPH.sub("₹", text) if "`" in text else text


def _clean_cell(cell: object) -> str:
    return fix_rupee(re.sub(r"\s+", " ", str(cell)).strip()) if cell is not None else ""


def clean_rows(raw: list[list[object]]) -> list[list[str]]:
    """Normalise cells, drop empty rows/columns, pad ragged rows."""
    rows = [[_clean_cell(c) for c in r] for r in raw]
    width = max((len(r) for r in rows), default=0)
    rows = [r + [""] * (width - len(r)) for r in rows if any(r)]
    keep = [i for i in range(width) if any(r[i] for r in rows)]
    return [[r[i] for i in keep] for r in rows]


def _horizontal_rules(page) -> list[tuple[float, float, float]]:
    """(x0, x1, y) of every horizontal rule drawn on the page: thin lines and hairline rectangles."""
    rules = []
    for drawing in page.get_drawings():
        for item in drawing["items"]:
            if item[0] == "l" and abs(item[1].y - item[2].y) < 1:
                rules.append((min(item[1].x, item[2].x), max(item[1].x, item[2].x), item[1].y))
            elif item[0] == "re" and item[1].height < 2 and item[1].width > 2:
                rules.append((item[1].x0, item[1].x1, item[1].y0))
    return rules


def _open_edges(page, found) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Missing outer borders of detected tables, as vertical lines to add.

    Many statements rule every row across all columns but draw no border on the outer edge, so the
    line-based detector stops at the last vertical line and drops the outermost column (the prior-year
    figures, typically). A side is "open" when words sit just beyond the table on at least a third of
    its rows AND the table's own horizontal rules carry on over them; the fix closes it at the end of
    those rules. Words beyond a table with no rules over them (a neighbouring text column) are left alone.
    """
    words = page.get_text("words")
    candidates = []
    for t in found:
        x0, y0, x1, y1 = t.bbox
        need = max(2, t.row_count / 3)
        inside = [w for w in words if y0 <= (w[1] + w[3]) / 2 <= y1]
        right = [w for w in inside if w[0] >= x1 - 1]
        left = [w for w in inside if w[2] <= x0 + 1]
        if len({round(w[1]) for w in right}) >= need or len({round(w[1]) for w in left}) >= need:
            candidates.append((t.bbox, need, right, left))
    if not candidates:
        return []
    rect = page.rect  # rules off the page (printers' crop marks) are not part of any table
    rules = [r for r in _horizontal_rules(page) if r[0] >= rect.x0 - 1 and r[1] <= rect.x1 + 1]
    lines = []
    for (x0, y0, x1, y1), need, right, left in candidates:
        near = [r for r in rules if y0 - 1 <= r[2] <= y1 + 1]
        x = _follow_rules(near, x1, +1)
        # Close the side only if that brings the words beyond it inside (else the redetection is wasted).
        if x > x1 + 5 and len({round(w[1]) for w in right if w[2] <= x + 1}) >= need:
            lines.append(((x, y0), (x, y1)))
        x = _follow_rules(near, x0, -1)
        if x < x0 - 5 and len({round(w[1]) for w in left if w[0] >= x - 1}) >= need:
            lines.append(((x, y0), (x, y1)))
    return lines


def _follow_rules(rules: list[tuple[float, float, float]], edge: float, direction: int) -> float:
    """How far the table's rules run on past `edge` (rightwards for +1, leftwards for -1).

    A rule counts if it reaches the edge (within 3pt) and runs on past it, whether drawn in one stroke or
    one cell at a time (segments are chained); at least two rows must carry one.
    """
    while True:
        if direction > 0:
            nxt = [r[1] for r in rules if r[0] <= edge + 3 and r[1] > edge + 1]
        else:
            nxt = [r[0] for r in rules if r[1] >= edge - 3 and r[0] < edge - 1]
        if len(nxt) < 2:
            return edge
        edge = max(nxt) if direction > 0 else min(nxt)


def find_page_tables(page, *, min_rows: int = 2, min_cols: int = 2) -> list[Table]:
    """Tables on a PyMuPDF page. Never raises: a failed detection just means no tables."""
    try:
        found = page.find_tables().tables
        if found:
            edges = _open_edges(page, found)
            if edges:  # redetect with the missing outer borders closed
                found = page.find_tables(add_lines=edges).tables or found
    except Exception as exc:  # find_tables can choke on odd vector content
        log.warning("table_detection_failed", page=page.number + 1, error=str(exc))
        return []
    tables = []
    for t in found:
        rows = clean_rows(t.extract())
        if len(rows) >= min_rows and rows and len(rows[0]) >= min_cols:
            tables.append(Table(bbox=tuple(t.bbox), rows=rows))
    return tables


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|")


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(_md_cell(c) for c in cells) + " |"


def _header_md(header: list[str]) -> str:
    return _md_row(header) + "\n" + "| " + " | ".join("---" for _ in header) + " |"


def table_to_markdown(rows: list[list[str]]) -> str:
    return "\n".join([_header_md(rows[0]), *(_md_row(r) for r in rows[1:])])


def table_to_markdown_chunks(rows: list[list[str]], max_tokens: int, title: str = "") -> list[str]:
    """Markdown pieces of at most ~max_tokens, split on row boundaries, header repeated in each.

    A non-empty `title` is repeated on its own line above every piece, so each piece says what it is
    (which statement, standalone or consolidated, which unit). A single row bigger than the budget is
    emitted on its own (rows are never cut mid-cell).
    """
    prefix = f"{title}\n" if title else ""
    header = _header_md(rows[0])
    budget = max_tokens - estimate_tokens(header) - estimate_tokens(prefix)
    pieces: list[str] = []
    cur: list[str] = []
    used = 0
    for row in rows[1:]:
        line = _md_row(row)
        t = estimate_tokens(line)
        if cur and used + t > budget:
            pieces.append(prefix + header + "\n" + "\n".join(cur))
            cur, used = [], 0
        cur.append(line)
        used += t
    if cur or not pieces:
        pieces.append(prefix + header + ("\n" + "\n".join(cur) if cur else ""))
    return pieces


# ---- unruled tables ---------------------------------------------------------------------------
# Some reports (HPCL's statements) draw no lines at all: the figures stand in right-aligned columns and
# find_tables sees nothing, or a scrap of note numbers. As plain text a statement flattens into one run
# ("Revenue From Operations Sale of Products 31 4,64,246.96 ... 2,098.69 1,822.19 4,66,345.65 ...") and the
# model reads the wrong figure for a line item. Rebuilt here from word positions: words are grouped into
# rows, figures into the columns their right edges line up in, the column headings ("2024-25") are the words
# just above each column, note references get their own column, and a subtotal printed without a label is
# named after its section ("Revenue From Operations (subtotal)"). Prose with numbers in it is rejected: in a
# table a figure stands apart from its label.

AMOUNT = re.compile(r"^\(?-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\)?$|^\(?-?\d+\.\d+\)?$")
NOTE_REF = re.compile(r"^\d{1,3}[A-Z]?(?:\([a-z0-9]+\))?$|^&$", re.IGNORECASE)
NIL = {"-", "–", "—"}
ROW_TOL = 3.0      # points: words whose centres are this close share a row
COL_TOL = 8.0      # points: right edges this close are one column
MIN_COL_ROWS = 6   # a figure column needs this many figures
MIN_ROWS = 8       # and the statement this many rows with figures
MIN_LABELLED = 0.6  # share of figure rows that carry a label (words)
MIN_GAP = 10.0     # points between a label and its first figure (prose has one space)
MIN_GAPPED = 0.8   # share of labelled figure rows with that gap
PROSE_WORDS = 12   # a row with more words than this and no figure is a line of prose
SEGMENT_GAP = 45.0  # points between rows that start a new table


def _is_amount(t: str) -> bool:
    return t in NIL or (bool(AMOUNT.match(t)) and any(ch in t for ch in ",."))


def _rows(words):
    rows: list[list] = []
    for w in sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0])):
        cy = (w[1] + w[3]) / 2
        if rows and abs(cy - rows[-1][0]) <= ROW_TOL:
            rows[-1][1].append(w)
        else:
            rows.append([cy, [w]])
    return [(cy, sorted(ws, key=lambda w: w[0])) for cy, ws in rows]


def _columns(rows):
    edges = sorted(w[2] for _, ws in rows for w in ws if _is_amount(w[4]))
    cols: list[list[float]] = []
    for x in edges:
        if cols and x - cols[-1][-1] <= COL_TOL:
            cols[-1].append(x)
        else:
            cols.append([x])
    return [(min(c), max(c)) for c in cols if len(c) >= MIN_COL_ROWS]


def find_text_tables(page, exclude: list[BBox] | tuple = ()) -> list[Table]:
    """Unruled tables on the page (see the section comment), top to bottom. Never raises."""
    try:
        return [Table(bbox=b, rows=r) for b, r in _text_tables(page, exclude)]
    except Exception as exc:  # a failed rebuild just means the page keeps its plain text
        log.warning("text_table_failed", page=page.number + 1, error=str(exc))
        return []


def _text_tables(page, exclude):
    def outside(w) -> bool:
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        return not any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in exclude)

    rows = _rows([w for w in page.get_text("words") if outside(w)])
    found = []
    for segment in _segments(rows):
        t = _text_table(segment)
        if t is not None:
            found.append(t)
    return found


def _segments(rows):
    """Rows split where a line of prose or a wide gap separates two tables (a note with a schedule above it
    and a second table below), so each gets its own columns and headings."""
    out: list[list] = [[]]
    prev_cy = None
    for cy, ws in rows:
        prose = len(ws) > PROSE_WORDS and not any(_is_amount(w[4]) for w in ws)
        if out[-1] and (prose or (prev_cy is not None and cy - prev_cy > SEGMENT_GAP)):
            out.append([])
        if not prose:
            out[-1].append((cy, ws))
        prev_cy = cy
    return [s for s in out if s]


def _text_table(rows):
    cols = _columns(rows)
    if len(cols) < 2:
        return None

    def col_of(w):
        for i, (lo, hi) in enumerate(cols):
            if lo - COL_TOL <= w[2] <= hi + COL_TOL:
                return i
        return None

    right_edge = max(hi for _, hi in cols) + COL_TOL  # side tabs ("Corporate Overview") sit beyond it
    data = []
    gaps = []  # per figure row: is the first figure set apart from the label (a column, not prose)?
    for cy, ws in rows:
        figs: dict[int, str] = {}
        label = []
        first_fig = None
        for w in ws:
            i = col_of(w) if _is_amount(w[4]) else None
            if i is not None:
                figs[i] = w[4] if i not in figs else figs[i] + " " + w[4]
                first_fig = w if first_fig is None else first_fig
            elif w[0] < right_edge:
                label.append(w)
        if figs and label:
            before = [w for w in label if w[2] <= first_fig[0] + 1]
            gaps.append(not before or first_fig[0] - max(w[2] for w in before) >= MIN_GAP)
        data.append((cy, label, figs))
    with_figs = [k for k, (_, _, f) in enumerate(data) if f]
    if len(with_figs) < MIN_ROWS:
        return None
    labelled = [k for k in with_figs if re.search(r"[A-Za-z]{3}", " ".join(w[4] for w in data[k][1]))]
    if len(labelled) < MIN_LABELLED * len(with_figs) or not gaps or sum(gaps) < MIN_GAPPED * len(gaps):
        return None  # prose with numbers in it, or a grid of bare figures
    first, last = with_figs[0], with_figs[-1]
    # Header: words above the first figure row that sit over a figure column ("2024-25", "As at ...").
    top_cy = data[first][0]
    head = [""] * len(cols)
    start = first
    for k in range(first - 1, -1, -1):
        cy, label, _ = data[k]
        if top_cy - cy > 70 or len(label) > PROSE_WORDS:
            break  # too far up, or a line of prose
        over = [w for w in label if any(w[0] >= lo - 70 and w[2] <= hi + COL_TOL for lo, hi in cols)]
        for i, (lo, hi) in enumerate(cols):
            here = " ".join(w[4] for w in over if w[0] >= lo - 70 and w[2] <= hi + COL_TOL)
            head[i] = (here + " " + head[i]).strip()
        if not over and start == k + 1:
            start = k  # a section line right above the first figures ("Revenue From Operations")
    if not any(head):
        return None  # figures with no column heading: not a statement
    out = [["Particulars", "Note", *head]]
    items: list[tuple[str, float | None]] = []  # (label, first figure) per row; no figure = a heading
    for _cy, label, figs in data[start:last + 1]:
        text, note = _split_note(label)
        if not figs:
            if text:
                out.append([text, note] + [""] * len(cols))
                items.append((text, None))
            continue
        value = _value(figs[min(figs)])
        if not re.search(r"[A-Za-z]", text):
            text = _subtotal_label(items, value)
            items.append(("", None))  # a subtotal closes its section
        else:
            items.append((text, value))
        out.append([text, note] + [figs.get(i, "") for i in range(len(cols))])
    if not any(r[1] for r in out[1:]):
        out = [[r[0], *r[2:]] for r in out]  # no note references: drop the column
    ws_all = [w for _, ws in rows[start:last + 1] for w in ws if w[0] < right_edge]
    bbox = (min(w[0] for w in ws_all), min(w[1] for w in ws_all), max(w[2] for w in ws_all),
            max(w[3] for w in ws_all))
    return bbox, [[fix_rupee(c) for c in r] for r in out]


def _value(fig: str) -> float | None:
    """'(2,716.21)' -> -2716.21, '-' -> 0.0; None if it is not one number."""
    t = fig.strip()
    if t in NIL:
        return 0.0
    neg = t.startswith("(") and t.endswith(")")
    try:
        v = float(t.strip("()").replace(",", ""))
    except ValueError:
        return None
    return -v if neg else v


def _subtotal_label(items: list[tuple[str, float | None]], value: float | None) -> str:
    """Name a figure printed without a label after the section heading whose items add up to it
    ("Revenue From Operations (subtotal)"). The sum is checked: a nearby heading is not enough, since a
    subtotal often closes a larger section than the last heading. No section adds up: "Subtotal"."""
    if value is not None:
        total = 0.0
        for label, v in reversed(items):
            if v is not None:
                total += v
            elif label == "":
                break  # an earlier subtotal: its items are already counted in it
            elif abs(total - value) <= max(0.015, abs(value) * 1e-6) and total != 0:
                return f"{label} (subtotal)"
    return "Subtotal"


def _split_note(label) -> tuple[str, str]:
    """'Sale of Products 31' -> ('Sale of Products', '31'): note references trail the label after a gap."""
    k = len(label)
    while k > 0 and NOTE_REF.match(label[k - 1][4]):
        k -= 1
    if 0 < k < len(label) and label[k][0] - label[k - 1][2] > 12:
        words, note = label[:k], label[k:]
    else:
        words, note = label, []
    return (re.sub(r"\s+", " ", " ".join(w[4] for w in words)).strip(), " ".join(w[4] for w in note))


def has_row_labels(table: Table) -> bool:
    """Whether the table's rows carry words. A detected grid of bare note numbers and figures does not."""
    return any(re.search(r"[A-Za-z]{3}", c) for r in table.rows[1:] for c in r)