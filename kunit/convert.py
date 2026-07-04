"""Conversion engine: scan pass (curve semantics, safety inventory) then
edit pass (field-preserving rescale)."""
from __future__ import annotations

import datetime
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Optional, Set, Tuple

from .parser import Block, KFile, STD8
from .schema import CUSTOM, SCAN_EXTRA, Spec, resolve
from .units import DIM_NAMES, Dim, UnitSystem, factor


class ConvertError(Exception):
    pass


class Ctx:
    def __init__(self, kf: KFile, src: Optional[UnitSystem],
                 dst: Optional[UnitSystem], opts: Optional[dict] = None):
        self.kf = kf
        self.src = src
        self.dst = dst
        self.opts = opts or {}
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.counts: Counter = Counter()
        self.curve_dims: Dict[int, Dict[Tuple[Dim, Dim], str]] = {}
        self.table_dims: Dict[int, Tuple[Dim, Dim, Dim]] = {}
        self.curve_blocks: Dict[int, List[Block]] = {}
        self.table_blocks: Dict[int, Block] = {}
        self.probes: Dict[str, list] = {"ro": [], "e": [], "d": [],
                                        "gravity_lcids": []}
        self.unknown: Dict[str, int] = {}
        self.soft: Dict[str, str] = {}
        self.hard: Dict[str, str] = {}
        self._fac: Dict[Dim, Fraction] = {}
        self.factors_used: Dict[Dim, Fraction] = {}

    def fac(self, dim: Dim) -> Fraction:
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

    def count(self, what: str) -> None:
        self.counts[what] += 1

    def register_curve(self, lcid: int, xdim: Dim, ydim: Dim, src: str) -> None:
        self.curve_dims.setdefault(lcid, {})[(xdim, ydim)] = src

    def register_table(self, tbid: int, vdim: Dim, xdim: Dim, ydim: Dim) -> None:
        self.table_dims[tbid] = (vdim, xdim, ydim)


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
    # static curve references
    if not edit:
        for (ci, fi, xdim, ydim) in spec.curves:
            if ci < len(data):
                v = kf.get_number(data[ci], STD8, block.long, fi)
                if v:
                    ctx.register_curve(int(v), xdim, ydim, block.name)
    if edit and (spec.cards or spec.repeat or spec.group):
        ctx.count(block.name)


def _walk(ctx: Ctx, edit: bool) -> None:
    for block in ctx.kf.blocks:
        kind, payload = resolve(block.name)
        if kind == "spec":
            _apply_spec(payload, block, ctx, edit)
            extra = SCAN_EXTRA.get(_base(block.name))
            if extra and not edit:
                extra(block, ctx, edit)
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


def _base(name: str) -> str:
    from .schema import _MAT_ALIASES
    for opt in ("_TITLE", "_ID"):
        if name.endswith(opt):
            name = name[: -len(opt)]
    return _MAT_ALIASES.get(name, name)


def scan(kf: KFile, src: Optional[UnitSystem]) -> Ctx:
    ctx = Ctx(kf, src, None)
    _walk(ctx, edit=False)
    return ctx


def convert(path: str, src: UnitSystem, dst: UnitSystem, out_path: str,
            blast_unit: Optional[int] = None,
            allow_unknown: bool = False) -> Ctx:
    kf = KFile(path)
    ctx = Ctx(kf, src, dst, {"blast_unit": blast_unit})
    _walk(ctx, edit=False)                      # pass 1: semantics + inventory

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

    _walk(ctx, edit=True)                       # pass 2: rewrite fields
    if ctx.errors:
        raise ConvertError("conversion errors:\n  " + "\n  ".join(ctx.errors))

    stamp = datetime.date.today().isoformat()
    hdr = [f"$ kunit: converted from {src.key} to {dst.key} on {stamp}",
           f"$ kunit: unit system is now  mass={dst.mass}  length={dst.length}"
           f"  time={dst.time}  ({dst.describe().split('(')[1]}"]
    kf.write(out_path, extra_header=hdr)
    return ctx


def report(ctx: Ctx, src: UnitSystem, dst: UnitSystem) -> str:
    out = [f"converted : {src.describe()}", f"       -> : {dst.describe()}", ""]
    if ctx.factors_used:
        out.append("factors applied:")
        for dim, f in sorted(ctx.factors_used.items(),
                             key=lambda kv: DIM_NAMES.get(kv[0], "")):
            out.append(f"  {DIM_NAMES.get(dim, dim):<18} x {float(f):.9G}")
    out.append("")
    out.append("keywords rescaled:")
    for k, n in sorted(ctx.counts.items()):
        out.append(f"  {k:<44} x{n}")
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
    if ctx.kf.max_fmt_err:
        out.append("")
        out.append(f"worst field-width rounding: {ctx.kf.max_fmt_err:.2E} "
                   "relative")
    return "\n".join(out)
