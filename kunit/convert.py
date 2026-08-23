"""Conversion engine: scan pass (curve semantics, safety inventory), post-scan
resolution (overrides, discrete-material DRO, table sub-curves), edit pass
(field-preserving rescale), then optional self-verification."""
from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from collections import Counter
from decimal import Decimal
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from .parser import Block, KFile, ParameterFieldError, STD8, parse_number

try:
    from .parser import FieldWidthError
except ImportError:  # parser owner defines this; guard until it lands
    class FieldWidthError(Exception):  # type: ignore[no-redef]
        line_idx: int
from .schema import (CUSTOM, EDIT_EXTRA, SCAN_EXTRA, Spec, _numint,
                     _strip_title, resolve)
from .units import (ANG_VEL, DIM_NAMES, DIMLESS, Dim, FORCE, LENGTH, MOMENT,
                    TEMP, UnitSystem, VELOCITY, factor)


class ConvertError(Exception):
    pass


class ScanResult:
    """Resolved semantics gathered in scan pass 1: curve/table axis
    dimensions, the blocks that define them, part->section->material links,
    discrete-material DRO flags, and detection probes."""

    def __init__(self) -> None:
        self.curve_dims: Dict[int, Dict[Tuple, str]] = {}
        self.table_dims: Dict[int, Tuple[Dim, Dim, Dim]] = {}
        self.curve_blocks: Dict[int, List[Tuple[KFile, Block]]] = {}
        self.table_blocks: Dict[int, Tuple[KFile, Block]] = {}
        self.table_pairs: Dict[int, List[int]] = {}
        self.table_nvalues: Dict[int, int] = {}
        self.part_links: List[Tuple[int, int, int]] = []
        self.sec_discrete_dro: Dict[int, int] = {}
        # S-ALE context guards, filled during the scan pass: *DEFINE_VECTOR
        # ids whose XT/YT/ZT are initial VELOCITIES rather than coordinates,
        # and *DEFINE_BOX ids whose six values are control-point INDICES
        # rather than coordinates (see h_ale_volume_filling).
        self.sale_vel_vectors: Set[int] = set()
        self.sale_index_boxes: Set[int] = set()
        # *CONTROL_SPH IDIM: 2D SPH (IDIM=2/-2) changes what the
        # *ELEMENT_SPH MASS field means (see h_element_sph).
        self.sph_idim: Optional[int] = None
        self.torsional_mats: Set[int] = set()
        self.smat_blocks: List[Tuple[KFile, Block, str]] = []
        # parameter-aware conversion (convert(..., parameters=True)):
        # NAME -> {factor: keyword that used it} gathered by the edit pass,
        # NAME -> [ParamDef] from collect_parameters, names defined by
        # *PARAMETER_EXPRESSION (never rescalable)
        self.param_factors: Dict[str, Dict[Fraction, str]] = {}
        self.param_use_counts: Counter = Counter()
        self.param_defs: Dict[str, List["ParamDef"]] = {}
        self.param_expr: Dict[str, list] = {}
        self.probes: Dict[str, list] = {"ro": [], "e": [], "d": [],
                                        "gravity_lcids": [],
                                        "gravity_accels": []}

    def register_curve(self, lcid: int, xdim, ydim, src: str) -> None:
        self.curve_dims.setdefault(lcid, {})[(xdim, ydim)] = src

    def register_table(self, tbid: int, vdim, xdim, ydim) -> None:
        self.table_dims[tbid] = (vdim, xdim, ydim)


class Ctx:
    """Conversion context: composes ``self.scan`` (the ScanResult holding all
    pass-1 resolved semantics) and holds the factor engine, the current-file
    ``kf`` pointer, diagnostics, and the edit-pass results."""

    def __init__(self, files: List[KFile], src: Optional[UnitSystem],
                 dst: Optional[UnitSystem], opts: Optional[dict] = None):
        # inputs
        self.files = files
        self.kf: Optional[KFile] = files[0] if files else None
        self.cur_block: str = ""
        self.src = src
        self.dst = dst
        self.opts = opts or {}
        # pass-1 resolved semantics
        self.scan = ScanResult()
        # factor engine
        self._fac: Dict[Dim, Fraction] = {}
        self.factors_used: Dict[Dim, Fraction] = {}
        # diagnostics
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.notes: List[str] = []
        self.counts: Counter = Counter()
        self.unknown: Dict[str, int] = {}
        self.soft: Dict[str, str] = {}
        self.hard: Dict[str, str] = {}
        # edit results
        self.written: List[Tuple[str, Optional[str]]] = []  # (out, backup)
        self.self_check: Optional[str] = None
        self.roundtrip: Optional[str] = None

    def fac(self, dim) -> Fraction:
        if dim is TEMP:
            return Fraction(1)          # classified, never rescaled
        f = self._fac.get(dim)
        if f is None:
            f = factor(dim, self.src, self.dst)
            self._fac[dim] = f
        if dim != (0, 0, 0):
            self.factors_used[dim] = f
        return f

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def error(self, msg: str) -> None:
        if msg not in self.errors:
            self.errors.append(msg)

    def note(self, msg: str) -> None:
        if msg not in self.notes:
            self.notes.append(msg)

    def count(self, what: str) -> None:
        self.counts[what] += 1


# ── *PARAMETER tables ────────────────────────────────────────────────────────

class ParamDef:
    """One ``R``/``I`` definition inside a *PARAMETER card: where the value
    field sits so it can be rescaled in place, and whether the definition is
    file-scoped (*PARAMETER_LOCAL, R16 Vol I p.36-4 Remark 5)."""
    __slots__ = ("name", "ptype", "kf", "line_idx", "fi", "value", "local")

    def __init__(self, name, ptype, kf, line_idx, fi, value, local=False):
        self.name, self.ptype, self.kf = name, ptype, kf
        self.line_idx, self.fi, self.value = line_idx, fi, value
        self.local = local


_PARAM_WIDTHS = [10] * 8


def _is_param_block(name: str) -> bool:
    return (name == "PARAMETER" or name.startswith("PARAMETER_")) and \
        "EXPRESSION" not in name and "DUPLICATION" not in name


def collect_parameters(files: List[KFile]):
    """Scan every *PARAMETER[_LOCAL|_MUTABLE|_NOECHO] card of the tree.

    Returns (table, defs, expr): table NAME -> Decimal value of the FIRST
    definition (the *PARAMETER_DUPLICATION default keeps the first), defs
    NAME -> every ParamDef (all of them are rescaled), and expr NAME ->
    [(kf, [line indices], type letter, free_format)] of every
    *PARAMETER_EXPRESSION definition (first line + continuation lines);
    free format is decided from the 10-char PRMR field alone, because the
    expression text itself may contain commas (max, min, sign, atan2, mod
    all take two arguments, p.36-9).  Layout per R16 Vol I *PARAMETER
    (p.36-2): 10-char PRMR = type letter + up to 9-char name (blanks
    ignored, case-insensitive) alternating with 10-char VAL, four pairs per
    card; comma lines follow the same name/value alternation.  Expression
    continuation lines leave the first 10 columns blank (p.36-8)."""
    table: Dict[str, Decimal] = {}
    defs: Dict[str, List[ParamDef]] = {}
    expr: Dict[str, List[Tuple[KFile, List[int], str]]] = {}
    for kf in files:
        for b in kf.blocks:
            if b.name.startswith("PARAMETER") and "EXPRESSION" in b.name:
                cur = None
                for li in b.data:
                    line = kf.lines[li]
                    # a fixed 10-column PRMR never contains a comma, so a
                    # comma there (or right after it) means free format -
                    # a comma later in the line is part of the expression
                    free = "," in line[:11]
                    head = (line.split(",", 1)[0] if free
                            else line[:10]).strip()
                    if not head:
                        if cur is not None:     # continuation (blank PRMR)
                            cur.append(li)
                        continue
                    name = head[1:].replace(" ", "").upper()
                    cur = [li]
                    expr.setdefault(name, []).append(
                        (kf, cur, head[:1].upper(), free))
                continue
            if not _is_param_block(b.name):
                continue
            for li in b.data:
                fl = kf.fields(li, _PARAM_WIDTHS, b.long)
                for fi in range(0, len(fl) - 1, 2):
                    prmr = fl[fi][0].strip()
                    if not prmr:
                        continue
                    ptype = prmr[0].upper()
                    name = prmr[1:].replace(" ", "").upper()
                    if ptype not in ("R", "I") or not name:
                        continue
                    v = parse_number(fl[fi + 1][0])
                    d = ParamDef(name, ptype, kf, li, fi + 1, v,
                                 local="LOCAL" in b.name)
                    defs.setdefault(name, []).append(d)
                    if v is not None and name not in table:
                        table[name] = v
    return table, defs, expr


# ── multi-file loading ───────────────────────────────────────────────────────

def load_tree(path: str, follow: bool, strict: bool = True):
    """Load a deck and (optionally) its *INCLUDE files, depth-first.

    Returns (files, inc_entries) where inc_entries is a list of
    (kf, line_idx, resolved_include_path) triples for rewriting references.
    """
    files: List[KFile] = []
    inc_entries: List[Tuple[KFile, int, str]] = []
    seen: Set[str] = set()

    def _load(p: str) -> None:
        cp = os.path.normcase(os.path.abspath(p))
        if cp in seen:
            return
        seen.add(cp)
        kf = KFile(p)
        files.append(kf)
        if not follow:
            return
        for b in kf.blocks:
            if b.name != "INCLUDE":
                continue
            for li in b.data:
                ref = kf.lines[li].strip()
                if not ref:
                    continue
                if ref.endswith("+"):
                    raise ConvertError(
                        f"{p}: continued *INCLUDE filename lines ('+') are "
                        "not supported - flatten or rename the include")
                rp = ref if os.path.isabs(ref) else os.path.join(
                    os.path.dirname(os.path.abspath(p)), ref)
                if not os.path.isfile(rp):
                    if strict:
                        raise ConvertError(f"{p}: include not found: {ref}")
                    continue
                inc_entries.append((kf, li, rp))
                _load(rp)

    _load(path)
    table, _defs, _expr = collect_parameters(files)
    for kf in files:
        kf.params = table
    return files, inc_entries


# ── walking ──────────────────────────────────────────────────────────────────

def _apply_spec(spec: Spec, block: Block, ctx: Ctx, edit: bool) -> None:
    kf = ctx.kf
    data = list(block.data)
    opts = block.name.split("_")
    if ("TITLE" in opts or "ID" in opts) and not (spec.group and spec.group[0].heading):
        data = data[1:]

    def do_card(card, li):
        if card.heading:
            return
        if edit:
            for fi, dim in card.dims.items():
                if dim is TEMP:
                    ctx.note(f"*{block.name}: temperature field left "
                             "unchanged (temperatures are never rescaled)")
                    continue
                kf.scale_field(li, card.widths, block.long, fi, ctx.fac(dim),
                               pad_right=card.pad_right.get(fi, 0))

    idx = 0
    for card in spec.cards:
        if idx >= len(data):
            break
        do_card(card, data[idx])
        idx += 1
    if spec.group:
        while idx < len(data):
            for card in spec.group:
                if idx >= len(data):
                    break
                do_card(card, data[idx])
                idx += 1
    elif spec.repeat is not None:
        while idx < len(data):
            do_card(spec.repeat, data[idx])
            idx += 1
    elif idx < len(data) and not spec.extra_ok and edit:
        ctx.warn(f"*{block.name}: {len(data) - idx} trailing card(s) beyond "
                 "the modelled layout left unscaled - verify manually.")
    if not edit:
        ncards = len(spec.cards)
        for (ci, fi, xdim, ydim) in spec.curves:
            if ci >= len(data):
                continue
            if ci >= ncards and spec.repeat is not None:
                rows = data[ci:]            # one curve ref per repeated card
            elif ci >= ncards and spec.group:
                rows = data[ci::len(spec.group)]   # per group repetition
            else:
                rows = [data[ci]]
            for li in rows:
                v = kf.get_number(li, STD8, block.long, fi)
                if v:
                    ctx.scan.register_curve(int(v), xdim, ydim, block.name)
    if edit and (spec.cards or spec.repeat or spec.group):
        ctx.count(block.name)


def _base(name: str) -> str:
    from .schema import _MAT_ALIASES
    for opt in ("_TITLE", "_ID"):
        if name.endswith(opt):
            name = name[: -len(opt)]
    return _MAT_ALIASES.get(name, name)


def _walk(ctx: Ctx, edit: bool) -> None:
    for kf in ctx.files:
        ctx.kf = kf
        for block in kf.blocks:
            if block.name == "INCLUDE" and ctx.opts.get("follow_includes"):
                continue
            ctx.cur_block = block.name
            kind, payload = resolve(block.name)
            if (kind == "hard" and ctx.opts.get("parameters")
                    and block.name.startswith("PARAMETER")):
                continue        # handled by _scale_parameters after the edit
            if kind == "spec":
                _apply_spec(payload, block, ctx, edit)
                extra = (EDIT_EXTRA if edit else SCAN_EXTRA).get(_base(block.name))
                if extra:
                    extra(block, ctx)
            elif kind == "custom":
                payload(block, ctx, edit)
            elif kind == "white":
                pass
            elif kind == "soft":
                ctx.soft[block.name] = payload
            elif kind == "hard":
                ctx.hard[block.name] = payload
            else:
                ctx.unknown[block.name] = ctx.unknown.get(block.name, 0) + 1


def _post_scan(ctx: Ctx) -> None:
    # CLI --curve overrides win over anything the scan derived
    for lcid, dims in (ctx.opts.get("curve_overrides") or {}).items():
        ctx.scan.curve_dims[lcid] = {tuple(dims): "--curve override"}

    # discrete materials: translational vs torsional via SECTION_DISCRETE DRO
    mid_dros: Dict[int, Set[int]] = {}
    for _pid, secid, mid in ctx.scan.part_links:
        dro = ctx.scan.sec_discrete_dro.get(secid)
        if dro is not None and mid:
            mid_dros.setdefault(mid, set()).add(dro)
    for mid, dros in mid_dros.items():
        if len(dros) > 1:
            ctx.warn(f"discrete material {mid} is used by both translational "
                     "and torsional parts - treated as TRANSLATIONAL; split "
                     "the material to convert correctly.")
        elif dros == {1}:
            ctx.scan.torsional_mats.add(mid)

    # spring/damper curve materials: dims depend on the torsional flag
    for kf, block, kind in ctx.scan.smat_blocks:
        data = _strip_title(block, list(block.data))
        if not data:
            continue
        mid = _numint(kf, data[0], STD8, block.long, 0)
        tors = mid in ctx.scan.torsional_mats
        xdim = DIMLESS if tors else LENGTH
        ydim = MOMENT if tors else FORCE
        rdim = ANG_VEL if tors else VELOCITY
        if kind == "S04":
            lcd = _numint(kf, data[0], STD8, block.long, 1)
            lcr = _numint(kf, data[0], STD8, block.long, 2)
            if lcd:
                ctx.scan.register_curve(lcd, xdim, ydim, f"MAT_S04 mid={mid}")
                ctx.scan.register_table(lcd, rdim, xdim, ydim)
            if lcr:
                ctx.scan.register_curve(lcr, rdim, DIMLESS, f"MAT_S04 mid={mid}")
        elif kind == "S05":
            lcdr = _numint(kf, data[0], STD8, block.long, 1)
            if lcdr:
                ctx.scan.register_curve(lcdr, rdim, ydim, f"MAT_S05 mid={mid}")
        elif kind == "S06":
            # R16 Vol II p.2-2034: LCDL / LCDU force (moment) vs
            # displacement (rotation); as tables the extra axis is velocity
            for fi, what in ((1, "LCDL"), (2, "LCDU")):
                lc = _numint(kf, data[0], STD8, block.long, fi)
                if lc:
                    ctx.scan.register_curve(lc, xdim, ydim, f"MAT_S06 {what} mid={mid}")
                    ctx.scan.register_table(lc, rdim, xdim, ydim)

    # tables: propagate axis dims to their sub-curves
    for tbid, (vdim, xdim, ydim) in list(ctx.scan.table_dims.items()):
        pairs = ctx.scan.table_pairs.get(tbid) or []
        if pairs:
            for lcid in pairs:
                ctx.scan.register_curve(lcid, xdim, ydim, f"DEFINE_TABLE {tbid}")
            continue
        loc = ctx.scan.table_blocks.get(tbid)
        if loc is None:
            continue
        kf, tblock = loc
        n = ctx.scan.table_nvalues.get(tbid, 0)
        bi = kf.blocks.index(tblock)
        got = 0
        for b in kf.blocks[bi + 1:]:
            # exact match: DEFINE_CURVE_SMOOTH / _FUNCTION are not the
            # plain sub-curves the table's values-following form expects
            if got >= n or b.name not in ("DEFINE_CURVE",
                                          "DEFINE_CURVE_TITLE"):
                break
            bdata = _strip_title(b, list(b.data))
            if bdata:
                lcid = _numint(kf, bdata[0], STD8, b.long, 0)
                if lcid:
                    ctx.scan.register_curve(lcid, xdim, ydim,
                                       f"DEFINE_TABLE {tbid} sub-curve")
                    got += 1
        if got < n:
            ctx.error(f"*DEFINE_TABLE {tbid}: expected {n} *DEFINE_CURVE "
                      f"blocks immediately following it (one per value), "
                      f"found {got} - sub-curve dimensions unresolved.")


def scan(files, src: Optional[UnitSystem], opts: Optional[dict] = None) -> Ctx:
    if isinstance(files, KFile):
        files = [files]
    ctx = Ctx(list(files), src, None, opts or {})
    _walk(ctx, edit=False)
    _post_scan(ctx)
    return ctx


def inventory(files, follow_includes: bool = False) -> Dict[str, Tuple[str, int]]:
    """Classify every keyword present: name -> (kind, occurrence count).

    kind is resolve()'s verdict: 'spec'/'custom' (scalable), 'white'
    (dimensionless), 'soft' (left unchanged with a warning), 'hard'
    (refused) or 'unknown'. *INCLUDE is skipped when follow_includes,
    mirroring convert()."""
    inv: Dict[str, Tuple[str, int]] = {}
    for kf in files:
        for block in kf.blocks:
            if block.name == "INCLUDE" and follow_includes:
                continue
            kind, _ = resolve(block.name)
            prev = inv.get(block.name)
            inv[block.name] = (kind, (prev[1] if prev else 0) + 1)
    return inv


# ── conversion ───────────────────────────────────────────────────────────────

def _scale_parameters(ctx: Ctx) -> None:
    """Rescale every *PARAMETER value that fed a dimensional field.

    Rules: a parameter must be used with ONE factor only (a density and a
    thickness sharing a name is a conflict), must be defined either by plain
    *PARAMETER cards or by *PARAMETER_EXPRESSION cards - not both (which
    definition wins depends on order and *PARAMETER_DUPLICATION, p.36-6) -
    and must be real-typed when the factor is not 1.  A name with several
    plain definitions is only rescaled when they are interchangeable (same
    file, same value, none LOCAL) - *PARAMETER_LOCAL gives the same name
    different meanings per file (p.36-4 Remark 5), and kunit's uses are
    keyed by name only.  After rescaling, every ``&name`` reference in the
    tree must have been seen by the factor sink; references in fields kunit
    does not scale (dimensionless fields, unknown keywords, curve scale
    factors) would silently receive the rescaled value and are refused.
    Violations are ctx.errors -> ConvertError."""
    for name, uses in sorted(ctx.scan.param_factors.items()):
        facs = set(uses)
        if len(facs) > 1:
            ctx.error(f"parameter &{name}: conflicting dimensions - used by "
                      + ", ".join(f"*{kw} (x{float(f):.6G})"
                                  for f, kw in sorted(uses.items(),
                                                      key=lambda kv: str(kv[1])))
                      + "; split the parameter to convert")
            continue
        f = next(iter(facs))
        if f == 1:
            continue
        if name in ctx.scan.param_expr and name in ctx.scan.param_defs:
            ctx.error(f"parameter &{name} is defined by both *PARAMETER and "
                      "*PARAMETER_EXPRESSION - which definition is active "
                      "depends on order and *PARAMETER_DUPLICATION (R16 "
                      "Vol I p.36-6); convert manually")
            continue
        if name in ctx.scan.param_expr and name not in ctx.scan.param_defs:
            _scale_expression(ctx, name, f, uses[f])
            continue
        defs = ctx.scan.param_defs.get(name)
        if not defs:
            ctx.error(f"parameter &{name} feeds a dimensional field "
                      f"(*{uses[f]}, x{float(f):.6G}) but is not defined in "
                      "the tree")
            continue
        if len(defs) > 1 and (any(d.local for d in defs)
                              or len({id(d.kf) for d in defs}) > 1
                              or len({d.value for d in defs}) > 1):
            ctx.error(f"parameter &{name} has {len(defs)} definitions with "
                      "*PARAMETER_LOCAL scoping, different values or in "
                      "different files - kunit resolves uses by name only "
                      "and cannot tell which definition each use sees (R16 "
                      "Vol I p.36-4 Remark 5); rename the parameter or "
                      "convert manually")
            continue
        for d in defs:
            if d.ptype == "I":
                ctx.error(f"parameter &{name} is an integer parameter but "
                          f"feeds a dimensional field (*{uses[f]}, "
                          f"x{float(f):.6G}) - cannot rescale")
                break
            if d.value is None:
                ctx.error(f"parameter &{name}: its *PARAMETER value is not "
                          "a number")
                break
            blk = next(b for b in d.kf.blocks if d.line_idx in b.data)
            d.kf.scale_field(d.line_idx, _PARAM_WIDTHS, blk.long, d.fi, f)
        else:
            ctx.count("PARAMETER")
            ctx.note(f"parameter &{name} rescaled x{float(f):.6G} "
                     f"(feeds *{uses[f]})")
    _check_unseen_references(ctx)


def _check_unseen_references(ctx: Ctx) -> None:
    """Refuse when a rescaled parameter is referenced in fields the edit
    pass never scaled: those fields (dimensionless Spec fields, unknown or
    soft keywords, unresolved curve scale factors, *INCLUDE names) would
    silently receive the rescaled value."""
    scaled = {n for n, u in ctx.scan.param_factors.items()
              if any(fx != 1 for fx in u)}
    if not scaled:
        return
    refs: Counter = Counter()
    ref_re = _re.compile(r"&\s*([A-Za-z_][A-Za-z0-9_]*)")
    for kf in ctx.files:
        for b in kf.blocks:
            if b.name.startswith("PARAMETER"):
                continue        # definitions and expression texts
            for li in b.data:
                line = kf.lines[li]
                if "&" in line:
                    for m in ref_re.finditer(line):
                        refs[m.group(1).upper()] += 1
    for name in sorted(scaled):
        extra = refs.get(name, 0) - ctx.scan.param_use_counts.get(name, 0)
        if extra > 0:
            ctx.error(f"parameter &{name} is rescaled but {extra} of its "
                      f"references sit in fields kunit does not scale "
                      "(dimensionless fields, unknown keywords, curve scale "
                      "factors or *INCLUDE names) - those would silently "
                      "receive the rescaled value; convert manually")


_re = __import__("re")
_IDENT_RE = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# numeric literals (1.0, .5, 2E-3, 1.0D+5) - blanked before scanning for
# identifiers so an exponent letter is not mistaken for a parameter name
_NUMLIT_RE = _re.compile(r"(?<![\w])(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?")
_EXPR_WIDTH = 70        # R16 Vol I p.36-7: PRMR in a field of 10, the rest


def _expr_text(kf: KFile, lines: List[int], free: bool) -> str:
    """Expression text of one *PARAMETER_EXPRESSION entry."""
    first = kf.lines[lines[0]]
    if free:
        return first.split(",", 1)[1].strip()
    parts = [first[10:].strip()] + [kf.lines[li].strip() for li in lines[1:]]
    return "".join(parts)


def _expr_idents(text: str) -> Set[str]:
    """Parameter names referenced by an expression: identifiers with numeric
    literals blanked first (so 1.0E-3 contributes no 'E') and function calls
    skipped (an identifier directly followed by '(' is one of the p.36-8
    functions, not a parameter)."""
    t = _NUMLIT_RE.sub(" ", text)
    out: Set[str] = set()
    for m in _IDENT_RE.finditer(t):
        j = m.end()
        while j < len(t) and t[j] == " ":
            j += 1
        if j < len(t) and t[j] == "(":
            continue
        out.add(m.group(0).upper())
    return out


def _fmt_factor(f: Fraction) -> str:
    """Factor as a plain decimal token (no exponent): the manual documents
    the expression grammar by example only, so 0.000001 is the safe spelling
    of 1e-6.  15 significant digits - below the 1e-9 field-format tolerance
    the self-check already accepts."""
    from decimal import localcontext
    with localcontext() as c:
        c.prec = 15
        d = (Decimal(f.numerator) / Decimal(f.denominator)).normalize()
    s = format(d, "f")
    return s if "." in s else s + ".0"


def _chunk_expr(text: str, width: int) -> Optional[List[str]]:
    """Split an expression into card-width chunks, breaking only after an
    operator, parenthesis or comma - the manual does not promise that a
    token may straddle continuation lines (R16 Vol I p.36-8).  A '+'/'-'
    that is part of an exponent (1.0E-3) is not a boundary.  Returns None
    when a chunk has no such boundary."""
    chunks: List[str] = []
    s = text
    while len(s) > width:
        cut = 0
        for p in range(width, 0, -1):
            c = s[p - 1]
            if c not in "+-*/(),":
                continue
            if c in "+-" and p >= 2 and s[p - 2] in "eEdD" \
                    and p < len(s) and s[p].isdigit():
                continue                      # exponent sign, not an operator
            cut = p
            break
        if not cut:
            return None
        chunks.append(s[:cut])
        s = s[cut:]
    chunks.append(s)
    return chunks


def _scale_expression(ctx: Ctx, name: str, f: Fraction, used_by: str) -> None:
    """Rescale a *PARAMETER_EXPRESSION result by wrapping it: ``expr`` ->
    ``(expr)*factor``.  Exact for the field it feeds; the expression's input
    parameters keep their original units (noted).  Refused when the result
    is itself an input of another expression (the wrapped value would leak
    into it) or when the wrapped text no longer fits its card (R16 Vol I
    p.36-8: continuation lines leave the first 10 columns blank)."""
    types = {t for _kf, _lines, t, _free in ctx.scan.param_expr[name]}
    if types - {"R"}:
        ctx.error(f"parameter &{name} is a *PARAMETER_EXPRESSION of type "
                  f"{'/'.join(sorted(types))} that feeds a dimensional field "
                  f"(*{used_by}, x{float(f):.6G}) - only real (R) "
                  "expressions can be rescaled (an integer result would be "
                  "truncated)")
        return
    # forward guard: if any input of this expression is itself rescaled
    # (a plain parameter feeding a dimensional field, or another wrapped
    # expression), LS-DYNA would re-evaluate the expression from the
    # rescaled inputs and the wrap would scale the result twice
    rescaled = {n for n, u in ctx.scan.param_factors.items()
                if any(fx != 1 for fx in u)}
    own_inputs: Set[str] = set()
    for kf, lines, _t, free in ctx.scan.param_expr[name]:
        own_inputs |= _expr_idents(_expr_text(kf, lines, free))
    bad_inputs = sorted((own_inputs & rescaled) - {name})
    if bad_inputs:
        ctx.error(f"parameter &{name} is a *PARAMETER_EXPRESSION whose "
                  f"input(s) " + ", ".join("&" + n for n in bad_inputs)
                  + " are themselves rescaled - wrapping the result would "
                  f"scale it twice (feeds *{used_by}, x{float(f):.6G}); "
                  "convert manually")
        return
    if name in own_inputs:
        ctx.error(f"parameter &{name} is a *PARAMETER_EXPRESSION redefined "
                  "in terms of itself (MUTABLE, R16 Vol I p.36-10 Remark 4) "
                  "- wrapping every definition would scale it twice; "
                  "convert manually")
        return
    # reverse guard: the wrapped result must not be an input of another
    # expression (which LS-DYNA would evaluate from the wrapped value)
    uses_in_expr = []
    for other, defs in ctx.scan.param_expr.items():
        if other == name:
            continue
        for kf, lines, _t, free in defs:
            if name in _expr_idents(_expr_text(kf, lines, free)):
                uses_in_expr.append(other)
    if uses_in_expr:
        ctx.error(f"parameter &{name} is a *PARAMETER_EXPRESSION result that "
                  f"feeds a dimensional field (*{used_by}, x{float(f):.6G}) "
                  f"and is also an input of the expression(s) "
                  + ", ".join("&" + u for u in sorted(set(uses_in_expr)))
                  + " - cannot rescale without double-scaling; convert "
                    "manually")
        return
    fac = _fmt_factor(f)
    for kf, lines, _t, free in ctx.scan.param_expr[name]:
        text = _expr_text(kf, lines, free)
        wrapped = f"({text})*{fac}"
        first = kf.lines[lines[0]]
        if free:
            head = first.split(",", 1)[0]
            new = f"{head},{wrapped}"
            if len(new) > 80:
                ctx.error(f"parameter &{name}: rescaled expression exceeds "
                          "80 columns - convert manually")
                return
            kf.lines[lines[0]] = new
            continue
        chunks = _chunk_expr(wrapped, _EXPR_WIDTH)
        if chunks is None:
            ctx.error(f"parameter &{name}: rescaled expression has no "
                      f"operator boundary within {_EXPR_WIDTH} columns to "
                      "continue the card at - convert manually")
            return
        if len(chunks) > len(lines):
            ctx.error(f"parameter &{name}: rescaled expression needs "
                      f"{len(chunks)} card lines but the deck has "
                      f"{len(lines)} - convert manually")
            return
        kf.lines[lines[0]] = first[:10] + chunks[0]
        for k, li in enumerate(lines[1:], 1):
            kf.lines[li] = " " * 10 + (chunks[k] if k < len(chunks) else "")
    ctx.count("PARAMETER_EXPRESSION")
    ctx.note(f"parameter expression &{name} wrapped as (expr)*{fac} (feeds "
             f"*{used_by}); its input parameters keep their original units")


def _out_path_for(in_path: str, dst: UnitSystem) -> str:
    stem, ext = os.path.splitext(in_path)
    return f"{stem}__{dst.key}{ext}"


def _note_stale_unit_comment(ctx: "Ctx", kf: KFile, src: UnitSystem) -> None:
    """Point out an author's 'Unit: ...' banner that the conversion made false.

    Comments are never rewritten (they may be a licence block or a hand
    edit), but leaving one behind silently is a trap for the next reader -
    and for kunit's own detector, which only ignores it because the provenance
    stamp outranks it."""
    from .detect import header_comment_system
    stale = header_comment_system(kf.lines[:80])
    if stale is not None and stale == src:
        ctx.note(f"the deck's own header comment still declares "
                 f"{src.key} - it is now stale; kunit's provenance stamp "
                 "above it records the real unit system")


def convert(path: str, src: UnitSystem, dst: UnitSystem, out_path: str,
            blast_unit: Optional[int] = None,
            allow_unknown: bool = False,
            follow_includes: bool = False,
            dry_run: bool = False,
            curve_overrides: Optional[Dict[int, Tuple[Dim, Dim]]] = None,
            self_check: bool = True,
            verify_roundtrip: bool = False,
            backup: bool = True,
            parameters: bool = False) -> Ctx:
    """parameters=True scales *PARAMETER values by the dimension of the
    fields that reference them (``&name``) instead of refusing the deck;
    see _scale_parameters for the rules."""
    files, inc_entries = load_tree(path, follow_includes)
    opts = {"blast_unit": blast_unit, "follow_includes": follow_includes,
            "curve_overrides": curve_overrides or {},
            "parameters": parameters}
    ctx = Ctx(files, src, dst, opts)
    if parameters:
        _table, ctx.scan.param_defs, ctx.scan.param_expr = \
            collect_parameters(files)

        def sink(name, f):
            ctx.scan.param_factors.setdefault(name, {}).setdefault(
                f, ctx.cur_block)
            ctx.scan.param_use_counts[name] += 1
        for kf in files:
            kf.param_sink = sink
    _walk(ctx, edit=False)                      # pass 1: semantics + inventory
    _post_scan(ctx)

    if ctx.hard:
        lines = [f"  *{k}: {v}" for k, v in sorted(ctx.hard.items())]
        raise ConvertError("keywords that cannot be safely converted:\n"
                           + "\n".join(lines))
    if ctx.unknown and not allow_unknown:
        lines = [f"  *{k} (x{n})" for k, n in sorted(ctx.unknown.items())]
        raise ConvertError(
            "unknown keywords (not classified as scalable or dimensionless):\n"
            + "\n".join(lines)
            + "\nRefusing to convert - their fields might be dimensional. "
              "Re-run with --allow-unknown to convert anyway (they will be "
              "left unchanged), or extend kunit/schema.py.")

    try:
        _walk(ctx, edit=True)                   # pass 2: rewrite fields
        if parameters:
            _scale_parameters(ctx)
    except (ParameterFieldError, FieldWidthError) as e:
        raise ConvertError(f"{ctx.kf.path}: {e}") from None
    if ctx.errors:
        raise ConvertError("conversion errors:\n  " + "\n  ".join(ctx.errors))

    # plan output paths (main file -> out_path; includes -> sibling __<to>)
    main_in = os.path.normcase(os.path.abspath(path))
    in_place = os.path.normcase(os.path.abspath(out_path)) == main_in
    plan: Dict[str, str] = {}                   # normcased input -> output
    for kf in files:
        cp = os.path.normcase(os.path.abspath(kf.path))
        if cp == main_in:
            plan[cp] = os.path.abspath(out_path)
        else:
            plan[cp] = (os.path.abspath(kf.path) if in_place
                        else _out_path_for(os.path.abspath(kf.path), dst))

    # rewrite *INCLUDE references to the converted filenames
    if not in_place:
        for kf, li, rp in inc_entries:
            new_base = os.path.basename(
                plan[os.path.normcase(os.path.abspath(rp))])
            ref = kf.lines[li].rstrip()
            head, tail = os.path.split(ref.strip())
            if ref.strip().endswith(tail):
                kf.lines[li] = ref[: len(ref) - len(tail)] + new_base

    if dry_run:
        ctx.note("DRY RUN - no files were written")
        return ctx

    stamp = datetime.date.today().isoformat()
    # plan backups first so a collision aborts BEFORE anything is
    # overwritten - never leave a half-converted include tree behind
    writes: List[Tuple[KFile, str, Optional[str]]] = []
    for kf in files:
        cp = os.path.normcase(os.path.abspath(kf.path))
        out = plan[cp]
        bak = None
        if os.path.normcase(out) == cp and backup:
            bak = out + f".orig_{src.key}"
            if os.path.exists(bak):
                raise ConvertError(f"backup {bak} already exists - refusing "
                                   "to overwrite it (delete or move it "
                                   "first); no files have been modified")
        writes.append((kf, out, bak))
    for kf, out, bak in writes:
        if bak:
            shutil.copy2(kf.path, bak)
        hdr = None
        if os.path.normcase(os.path.abspath(kf.path)) == main_in:
            hdr = [f"$ kunit: converted from {src.key} to {dst.key} on {stamp}",
                   f"$ kunit: unit system is now  mass={dst.mass}  "
                   f"length={dst.length}  time={dst.time}"]
        # write to a sibling temp file then atomically replace the target,
        # so a mid-tree failure never leaves a partially-overwritten file
        tmp = out + ".tmp_kunit"
        try:
            kf.write(tmp, extra_header=hdr)
            os.replace(tmp, out)
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        ctx.written.append((out, bak))
        if cp == main_in:
            _note_stale_unit_comment(ctx, kf, src)

    # self-check: the output should auto-detect as the target system from
    # PHYSICAL evidence alone - header comments (including the kunit stamp
    # this very conversion just wrote) are ignored, otherwise the check
    # would confirm its own claim and could never fail
    if self_check:
        from .detect import detect
        try:
            v = detect(out_path, follow_includes=follow_includes,
                       use_headers=False)
            if v.system is None:
                ctx.self_check = "no evidence in output - self-check skipped"
            elif v.system == dst:
                ctx.self_check = ("OK - output auto-detects as "
                                  + dst.key
                                  + (" (ambiguous score)" if v.ambiguous else ""))
            else:
                ctx.self_check = f"FAILED - output detects as {v.system.key}"
                ctx.warn(f"SELF-CHECK FAILED: the converted deck auto-detects "
                         f"as {v.system.key}, not {dst.key}. A dimensional "
                         "field was probably missed - inspect the output!")
        except Exception as e:                  # detection must never break conversion
            ctx.self_check = f"error: {e}"

    if verify_roundtrip:
        n_wrapped = ctx.counts.get("PARAMETER_EXPRESSION", 0)
        if n_wrapped:
            ctx.roundtrip = (f"skipped ({n_wrapped} *PARAMETER_EXPRESSION "
                             "wrapped - re-wrapping is not idempotent)")
        elif len(files) > 1:
            ctx.roundtrip = "skipped (multi-file tree)"
        else:
            ctx.roundtrip = _roundtrip(out_path, src, dst, blast_unit,
                                       curve_overrides, parameters)
            if not ctx.roundtrip.startswith("OK"):
                ctx.warn("ROUNDTRIP CHECK: " + ctx.roundtrip)
    return ctx


def _roundtrip(out_path: str, src: UnitSystem, dst: UnitSystem,
               blast_unit, curve_overrides, parameters: bool = False) -> str:
    """Convert output back to src and forward again; the two forward results
    must agree byte-for-byte (comments ignored) or precision was lost.
    (A rescaled *PARAMETER_EXPRESSION is wrapped again on every pass, so
    decks with such expressions report a textual difference by design.)"""
    tmp = tempfile.mkdtemp(prefix="kunit_rt_")
    back = os.path.join(tmp, "back.k")
    fwd2 = os.path.join(tmp, "fwd2.k")
    try:
        convert(out_path, dst, src, back, blast_unit=blast_unit,
                curve_overrides=curve_overrides, allow_unknown=True,
                self_check=False, parameters=parameters)
        convert(back, src, dst, fwd2, blast_unit=blast_unit,
                curve_overrides=curve_overrides, allow_unknown=True,
                self_check=False, parameters=parameters)

        def payload(p):
            with open(p, newline="") as fh:
                return [ln for ln in fh.read().splitlines()
                        if not ln.lstrip().startswith("$")]

        a, b = payload(out_path), payload(fwd2)
        if len(a) != len(b):
            return f"FAILED - line counts differ ({len(a)} vs {len(b)})"
        bad = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
        if bad:
            return (f"FAILED - {len(bad)} line(s) not reproduced, first at "
                    f"payload line {bad[0] + 1}: {a[bad[0]]!r} vs {b[bad[0]]!r}")
        return f"OK - {len(a)} payload lines reproduced exactly"
    except ConvertError as e:
        return f"FAILED - back-conversion error: {e}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def report(ctx: Ctx, src: UnitSystem, dst: UnitSystem) -> str:
    out = [f"converted : {src.describe()}", f"       -> : {dst.describe()}", ""]
    if ctx.factors_used:
        out.append("factors applied:")
        for dim, f in sorted(ctx.factors_used.items(),
                             key=lambda kv: str(DIM_NAMES.get(kv[0], ""))):
            out.append(f"  {str(DIM_NAMES.get(dim, dim)):<28} x {float(f):.9G}")
    out.append("")
    out.append("keywords rescaled:")
    for k, n in sorted(ctx.counts.items()):
        out.append(f"  {k:<44} x{n}")
    if ctx.written:
        out.append("")
        out.append("files written:")
        for w, bak in ctx.written:
            out.append(f"  {w}" + (f"   (backup: {bak})" if bak else ""))
    if ctx.self_check:
        out.append("")
        out.append(f"self-check : {ctx.self_check}")
    if ctx.roundtrip:
        out.append(f"roundtrip  : {ctx.roundtrip}")
    if ctx.notes:
        out.append("")
        out.append("notes:")
        for n in ctx.notes:
            out.append(f"  - {n}")
    if ctx.soft:
        out.append("")
        out.append("left unchanged (assumed dimensionless - verify):")
        for k, why in sorted(ctx.soft.items()):
            out.append(f"  *{k}: {why}")
    if ctx.unknown:
        out.append("")
        out.append("UNKNOWN keywords left unchanged (--allow-unknown):")
        for k, n in sorted(ctx.unknown.items()):
            out.append(f"  *{k} (x{n})")
    if ctx.warnings:
        out.append("")
        out.append(f"warnings ({len(ctx.warnings)}):")
        for w in ctx.warnings:
            out.append(f"  - {w}")
    fmt_err = max((kf.max_fmt_err for kf in ctx.files), default=0.0)
    if fmt_err:
        out.append("")
        out.append(f"worst field-width rounding: {fmt_err:.2E} relative")
    return "\n".join(out)
