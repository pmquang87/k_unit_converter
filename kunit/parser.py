"""Line-preserving LS-DYNA keyword file model.

The file is kept as a list of lines; blocks only reference line indices, and
edits rewrite single fields in place so untouched bytes survive verbatim.
Handles fixed-width cards (per-card column layouts), comma free-format lines,
E-less Fortran exponents (`7.85000-9`) and the long (`+` / LONG=Y) format.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import List, Optional, Sequence, Tuple

_ELESS = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+))([+-]\d+)$")

STD8 = [10] * 8  # default LS-DYNA card: 8 fields x 10 chars


def parse_number(token: str) -> Optional[Decimal]:
    t = token.strip()
    if not t:
        return None
    m = _ELESS.match(t)
    if m:
        t = m.group(1) + "E" + m.group(2)
    elif "D" in t or "d" in t:
        # Fortran double exponents (1.0D+5) - written by some exporters
        # and accepted by LS-DYNA
        t = t.replace("D", "E").replace("d", "e")
    try:
        v = Decimal(t)
    except InvalidOperation:
        return None
    # Decimal accepts 'nan'/'inf' - never treat those as field values
    return v if v.is_finite() else None


def format_fixed(value: Decimal, width: int, pad_right: int = 0):
    """Shortest float representation of `value` fitting `width` chars.

    Returns (text, rel_err). pad_right shifts the value left off the field's
    right edge (still spec-conformant for F-fields, which are parsed by
    whitespace-stripping the whole column range).
    """
    inner = width - pad_right
    if value == 0:
        s = "0.0" if inner >= 3 else "0"
        return s.rjust(inner) + " " * pad_right, 0.0
    f = float(value)
    best = None
    for p in range(12, 0, -1):
        s = f"{f:.{p}G}"
        if len(s) <= inner:
            best = s
            break
    if best is None:
        raise ValueError(f"cannot format {value} in {inner} chars")
    rel = abs(float(best) / f - 1.0) if f != 0 else 0.0
    return best.rjust(inner) + " " * pad_right, rel


@dataclass
class Block:
    name: str                 # keyword without '*', upper-case
    kwline: int               # index of the '*KEYWORD' line
    data: List[int] = field(default_factory=list)  # data line indices
    long: bool = False

    def options(self) -> List[str]:
        return self.name.split("_")


class KFile:
    def __init__(self, path: str):
        self.path = path
        with open(path, "r", newline="") as fh:
            text = fh.read()
        self.eol = "\r\n" if "\r\n" in text[:20000] else "\n"
        self.lines: List[str] = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        self.blocks: List[Block] = self._scan_blocks()
        self.max_fmt_err = 0.0

    def _scan_blocks(self) -> List[Block]:
        blocks: List[Block] = []
        cur: Optional[Block] = None
        global_long = False
        for i, ln in enumerate(self.lines):
            if ln.startswith("*"):
                raw = ln.strip()
                name = raw.lstrip("*")
                long = global_long
                if name.endswith("+"):
                    long, name = True, name[:-1].strip()
                elif name.endswith("-"):
                    long, name = False, name[:-1].strip()
                # the keyword is the first whitespace token: trailing text like
                # '*KEYWORD MEMORY=... NCPU=4' or LS-PrePost's
                # '*ELEMENT_SOLID (TEN NODES FORMAT)' is not part of the name
                tokens = name.strip().upper().split()
                if tokens and tokens[0] == "+":
                    long, tokens = True, tokens[1:]
                name = tokens[0] if tokens else ""
                if name == "KEYWORD" and "LONG" in raw.upper():
                    m = re.search(r"LONG\s*=\s*([A-Z])", raw.upper())
                    if m and m.group(1) == "Y":
                        global_long = True
                cur = Block(name=name, kwline=i, long=long)
                blocks.append(cur)
            elif cur is not None and not ln.lstrip().startswith("$"):
                # blank lines ARE data cards in LS-DYNA (all fields default,
                # commonly used to skip an optional card); dropping them
                # would shift every later card onto the wrong dimension map
                cur.data.append(i)
        for b in blocks:
            # cosmetic trailing blanks (incl. the empty final line a
            # trailing newline produces) are not cards
            while b.data and not self.lines[b.data[-1]].strip():
                b.data.pop()
        return blocks

    # ── field access ────────────────────────────────────────────────────────
    def is_free(self, line_idx: int) -> bool:
        return "," in self.lines[line_idx]

    def fields(self, line_idx: int, widths: Sequence[int], long: bool
               ) -> List[Tuple[str, int, int]]:
        """(text, start, end) triples. For free-format lines start is the
        token index and end is -1."""
        line = self.lines[line_idx]
        if self.is_free(line_idx):
            return [(tok, k, -1) for k, tok in enumerate(line.split(","))]
        if long:
            widths = [20] * len(widths)
        out, pos = [], 0
        for w in widths:
            out.append((line[pos:pos + w], pos, pos + w))
            pos += w
        return out

    def get_number(self, line_idx: int, widths: Sequence[int], long: bool,
                   fi: int) -> Optional[Decimal]:
        fl = self.fields(line_idx, widths, long)
        if fi >= len(fl):
            return None
        return parse_number(fl[fi][0])

    def set_field(self, line_idx: int, widths: Sequence[int], long: bool,
                  fi: int, text_value: str) -> None:
        """Replace one field with a pre-formatted string (fixed) or token (free)."""
        line = self.lines[line_idx]
        if self.is_free(line_idx):
            toks = line.split(",")
            while len(toks) <= fi:
                toks.append("")
            toks[fi] = text_value.strip()
            self.lines[line_idx] = ",".join(toks)
            return
        if long:
            widths = [20] * len(widths)
        tv = text_value.rjust(widths[fi])
        if len(tv) > widths[fi]:
            raise ValueError(f"value {text_value!r} does not fit the "
                             f"{widths[fi]}-char field {fi}")
        start = sum(widths[:fi])
        end = start + widths[fi]
        if len(line) < end:
            line = line.ljust(end)
        self.lines[line_idx] = line[:start] + tv + line[end:]

    def scale_field(self, line_idx: int, widths: Sequence[int], long: bool,
                    fi: int, f, pad_right: int = 0) -> bool:
        """Multiply field fi by Fraction f in place. Returns True if changed."""
        from .units import apply_factor
        fl = self.fields(line_idx, widths, long)
        if fi >= len(fl):
            return False
        raw = fl[fi][0]
        if "&" in raw:
            raise ParameterFieldError(line_idx, raw.strip())
        v = parse_number(raw)
        if v is None or v == 0 or f == 1:
            return False
        nv = apply_factor(v, f)
        if self.is_free(line_idx):
            s = f"{float(nv):.9G}"
            self.set_field(line_idx, widths, long, fi, s)
        else:
            w = 20 if long else widths[fi]
            s, rel = format_fixed(nv, w, pad_right)
            self.max_fmt_err = max(self.max_fmt_err, rel)
            line = self.lines[line_idx]
            start = sum(([20] * len(widths) if long else widths)[:fi])
            end = start + w
            if len(line) < end:
                line = line.ljust(end)
            self.lines[line_idx] = line[:start] + s + line[end:]
        return True

    # ── output ──────────────────────────────────────────────────────────────
    def write(self, path: str, extra_header: Optional[List[str]] = None) -> None:
        lines = self.lines
        if extra_header:
            for b in self.blocks:
                if b.name == "KEYWORD":
                    at = b.kwline + 1
                    lines = lines[:at] + extra_header + lines[at:]
                    break
            else:
                lines = extra_header + lines
        with open(path, "w", newline="") as fh:
            fh.write(self.eol.join(lines) if self.eol != "\n" else "\n".join(lines))


class ParameterFieldError(Exception):
    def __init__(self, line_idx: int, token: str):
        super().__init__(f"line {line_idx + 1}: field {token!r} references a "
                         f"*PARAMETER — cannot scale parametrised values")
        self.line_idx = line_idx
