"""Conversion engine: scan pass (curve semantics, safety inventory), post-scan
resolution (overrides, discrete-material DRO, table sub-curves), edit pass
(field-preserving rescale), then optional self-verification."""
from __future__ import annotations

import datetime
import os
import shutil
import tempfile
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from .parser import Block, KFile, ParameterFieldError, STD8
from .schema import (CUSTOM, EDIT_EXTRA, SCAN_EXTRA, Spec, _numint,
                     _strip_title, resolve)
from .units import (ANG_VEL, DIM_NAMES, DIMLESS, Dim, FORCE, LENGTH, MOMENT,
                    TEMP, UnitSystem, VELOCITY, factor)


class ConvertError(Exception):
    pass


class Ctx:
    def __init__(self, files: List[KFile], src: Optional[UnitSystem],
                 dst: Optional[UnitSystem], opts: Optional[dict] = None):
        self.files = files
        self.kf: KFile = files[0] if files else None
        self.src = src
        self.dst = dst
        self.opts = opts or {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.notes: List[str] = []
        self.counts: Counter = Counter()
        self.curve_dims: Dict[int, Dict[Tuple, str]] = {}
        self.table_dims: Dict[int, Tuple[Dim, Dim, Dim]] = {}
        self.curve_blocks: Dict[int, List[Tuple[KFile, Block]]] = {}
        self.table_blocks: Dict[int, Tuple[KFile, Block]] = {}
        self.table_pairs: Dict[int, List[int]] = {}
        self.table_nvalues: Dict[int, int] = {}
        self.part_links: List[Tuple[int, int, int]] = []
        self.sec_discrete_dro: Dict[int, int] = {}
        self.torsional_mats: Set[int] = set()
        self.smat_blocks: List[Tuple[KFile, Block, str]] = []
        self.probes: Dict[str, list] = {"ro": [], "e": [], "d": [],
                                        "gravity_lcids": [],
                                        "gravity_accels": []}
        self.unknown: Dict[str, int] = {}
        self.soft: Dict[str, str] = {}
        self.hard: Dict[str, str] = {}
        self._fac: Dict[Dim, Fraction] = {}
        self.factors_used: Dict[Dim, Fraction] = {}
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

    def register_curve(self, lcid: int, xdim, ydim, src: str) -> None:
        self.curve_dims.setdefault(lcid, {})[(xdim, ydim)] = src

    def register_table(self, tbid: int, vdim, xdim, ydim) -> None:
        self.table_dims[tbid] = (vdim, xdim, ydim)


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
                    ctx.register_curve(int(v), xdim, ydim, block.name)
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
            kind, payload = resolve(block.name)
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
        ctx.curve_dims[lcid] = {tuple(dims): "--curve override"}

    # discrete materials: translational vs torsional via SECTION_DISCRETE DRO
    mid_dros: Dict[int, Set[int]] = {}
    for _pid, secid, mid in ctx.part_links:
        dro = ctx.sec_discrete_dro.get(secid)
        if dro is not None and mid:
            mid_dros.setdefault(mid, set()).add(dro)
    for mid, dros in mid_dros.items():
        if len(dros) > 1:
            ctx.warn(f"discrete material {mid} is used by both translational "
                     "and torsional parts - treated as TRANSLATIONAL; split "
                     "the material to convert correctly.")
        elif dros == {1}:
            ctx.torsional_mats.add(mid)

    # spring/damper curve materials: dims depend on the torsional flag
    for kf, block, kind in ctx.smat_blocks:
        data = _strip_title(block, list(block.data))
        if not data:
            continue
        mid = _numint(kf, data[0], STD8, block.long, 0)
        tors = mid in ctx.torsional_mats
        xdim = DIMLESS if tors else LENGTH
        ydim = MOMENT if tors else FORCE
        rdim = ANG_VEL if tors else VELOCITY
        if kind == "S04":
            lcd = _numint(kf, data[0], STD8, block.long, 1)
            lcr = _numint(kf, data[0], STD8, block.long, 2)
            if lcd:
                ctx.register_curve(lcd, xdim, ydim, f"MAT_S04 mid={mid}")
                ctx.register_table(lcd, rdim, xdim, ydim)
            if lcr:
                ctx.register_curve(lcr, rdim, DIMLESS, f"MAT_S04 mid={mid}")
        elif kind == "S05":
            lcdr = _numint(kf, data[0], STD8, block.long, 1)
            if lcdr:
                ctx.register_curve(lcdr, rdim, ydim, f"MAT_S05 mid={mid}")

    # tables: propagate axis dims to their sub-curves
    for tbid, (vdim, xdim, ydim) in list(ctx.table_dims.items()):
        pairs = ctx.table_pairs.get(tbid) or []
        if pairs:
            for lcid in pairs:
                ctx.register_curve(lcid, xdim, ydim, f"DEFINE_TABLE {tbid}")
            continue
        loc = ctx.table_blocks.get(tbid)
        if loc is None:
            continue
        kf, tblock = loc
        n = ctx.table_nvalues.get(tbid, 0)
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
                    ctx.register_curve(lcid, xdim, ydim,
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

def _out_path_for(in_path: str, dst: UnitSystem) -> str:
    stem, ext = os.path.splitext(in_path)
    return f"{stem}__{dst.key}{ext}"


def convert(path: str, src: UnitSystem, dst: UnitSystem, out_path: str,
            blast_unit: Optional[int] = None,
            allow_unknown: bool = False,
            follow_includes: bool = False,
            dry_run: bool = False,
            curve_overrides: Optional[Dict[int, Tuple[Dim, Dim]]] = None,
            self_check: bool = True,
            verify_roundtrip: bool = False,
            backup: bool = True) -> Ctx:
    files, inc_entries = load_tree(path, follow_includes)
    opts = {"blast_unit": blast_unit, "follow_includes": follow_includes,
            "curve_overrides": curve_overrides or {}}
    ctx = Ctx(files, src, dst, opts)
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
    except ParameterFieldError as e:
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
        kf.write(out, extra_header=hdr)
        ctx.written.append((out, bak))

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
        if len(files) > 1:
            ctx.roundtrip = "skipped (multi-file tree)"
        else:
            ctx.roundtrip = _roundtrip(out_path, src, dst, blast_unit,
                                       curve_overrides)
            if not ctx.roundtrip.startswith("OK"):
                ctx.warn("ROUNDTRIP CHECK: " + ctx.roundtrip)
    return ctx


def _roundtrip(out_path: str, src: UnitSystem, dst: UnitSystem,
               blast_unit, curve_overrides) -> str:
    """Convert output back to src and forward again; the two forward results
    must agree byte-for-byte (comments ignored) or precision was lost."""
    tmp = tempfile.mkdtemp(prefix="kunit_rt_")
    back = os.path.join(tmp, "back.k")
    fwd2 = os.path.join(tmp, "fwd2.k")
    try:
        convert(out_path, dst, src, back, blast_unit=blast_unit,
                curve_overrides=curve_overrides, allow_unknown=True,
                self_check=False)
        convert(back, src, dst, fwd2, blast_unit=blast_unit,
                curve_overrides=curve_overrides, allow_unknown=True,
                self_check=False)

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
                             key=lambda kv: DIM_NAMES.get(kv[0], "")):
            out.append(f"  {DIM_NAMES.get(dim, dim):<28} x {float(f):.9G}")
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
