"""kunit command line interface.

  kunit detect  deck.k [--json]
  kunit check   deck.k [--follow-includes] [--json]
  kunit convert deck.k --to ton-mm-s [--from kg-m-s] [-o out.k]
  kunit convert deck.k --to ton-mm-s --in-place --follow-includes
  kunit systems
  kunit gui
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .convert import ConvertError, convert, inventory, load_tree, report, scan
from .detect import detect
from .units import DIM_NAMES, PRESETS, parse_dim_name, parse_system


def cmd_systems(_args) -> int:
    print("preset unit systems (custom MASS-LENGTH-TIME triples also accepted):")
    for s in PRESETS.values():
        print(f"  {s.describe()}")
    return 0


def cmd_detect(args) -> int:
    v = detect(args.deck, follow_includes=not args.no_includes)
    if args.json:
        print(json.dumps({
            "deck": args.deck,
            "system": v.system.key if v.system else None,
            "ambiguous": v.ambiguous,
            "scores": [{"system": s.key, "score": sc} for sc, s in v.ranked],
            "evidence": v.evidence,
        }, indent=2))
        return 2 if v.system is None else (1 if v.ambiguous else 0)
    print(v.table())
    if v.evidence:
        print("\nevidence:")
        for e in v.evidence:
            print(f"  - {e}")
    if v.system is None:
        print("\nno usable evidence found - specify units manually.")
        return 2
    print(f"\ndetected: {v.system.describe()}"
          + ("   [AMBIGUOUS - verify!]" if v.ambiguous else ""))
    return 1 if v.ambiguous else 0


def _dim_label(dim) -> str:
    return DIM_NAMES.get(dim, str(dim))


def cmd_check(args) -> int:
    """Coverage / convertibility report: classify every keyword and every
    *DEFINE_CURVE without touching the deck."""
    try:
        files, _inc = load_tree(args.deck, args.follow_includes)
    except ConvertError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    ctx = scan(files, None, {"follow_includes": args.follow_includes})
    inv = inventory(files, follow_includes=args.follow_includes)

    kinds = {"spec": [], "custom": [], "white": [], "soft": [], "hard": [],
             "unknown": []}
    for name, (kind, n) in sorted(inv.items()):
        kinds[kind].append((name, n))

    curves = {}
    for lcid in sorted(k for k in set(ctx.curve_blocks) | set(ctx.curve_dims)
                       if k is not None):
        dims = ctx.curve_dims.get(lcid, {})
        demands = [{"x": _dim_label(xd), "y": _dim_label(yd), "by": src}
                   for (xd, yd), src in sorted(dims.items(), key=str)]
        status = ("unresolved" if not dims
                  else "conflict" if len(dims) > 1 else "resolved")
        curves[lcid] = {"defined": lcid in ctx.curve_blocks,
                        "status": status, "demands": demands}

    hard = {k: v for k, v in ctx.hard.items()}
    unknown = dict(ctx.unknown)
    convertible = not hard and not unknown
    rc = 1 if hard else (2 if unknown else 0)

    if args.json:
        print(json.dumps({
            "deck": args.deck,
            "files": [kf.path for kf in files],
            "keywords": {name: {"kind": kind, "count": n}
                         for name, (kind, n) in sorted(inv.items())},
            "hard": hard,
            "soft": dict(ctx.soft),
            "unknown": unknown,
            "curves": {str(k): v for k, v in curves.items()},
            "warnings": ctx.warnings,
            "convertible": convertible,
        }, indent=2))
        return rc

    print(f"deck: {args.deck}" + (f"  ({len(files)} files, includes followed)"
                                  if len(files) > 1 else ""))
    label = {"spec": "scalable (dimension table)",
             "custom": "scalable (custom handler)",
             "white": "dimensionless (whitelist)",
             "soft": "left unchanged (assumed dimensionless - verify)",
             "hard": "HARD STOPS (conversion refused)",
             "unknown": "UNKNOWN (refused unless --allow-unknown)"}
    for kind in ("spec", "custom", "white", "soft", "hard", "unknown"):
        if not kinds[kind]:
            continue
        print(f"\n{label[kind]}:")
        for name, n in kinds[kind]:
            why = (ctx.hard.get(name) if kind == "hard"
                   else ctx.soft.get(name) if kind == "soft" else None)
            print(f"  *{name:<42} x{n}" + (f"   ({why})" if why else ""))
    if curves:
        print("\ncurves (*DEFINE_CURVE):")
        for lcid, c in curves.items():
            if c["status"] == "resolved":
                d = c["demands"][0]
                print(f"  lcid {lcid:<6} {d['x']} vs {d['y']}   [{d['by']}]")
            elif c["status"] == "conflict":
                wants = "; ".join(f"{d['x']}:{d['y']} ({d['by']})"
                                  for d in c["demands"])
                print(f"  lcid {lcid:<6} CONFLICT - {wants}")
            else:
                where = ("" if c["defined"]
                         else " (referenced but not defined here)")
                print(f"  lcid {lcid:<6} UNRESOLVED - no referencing keyword "
                      f"declares its dimensions{where}; use "
                      f"--curve {lcid}=<xdim>:<ydim> when converting")
    for w in ctx.warnings:
        print(f"\nwarning: {w}")
    print("\nverdict: " + (
        "convertible - every keyword is classified" if convertible
        else "NOT convertible - hard stops present" if hard
        else "needs --allow-unknown (or a schema extension) - unknown "
             "keywords present"))
    return rc


def parse_curve_overrides(specs):
    out = {}
    for s in specs or []:
        try:
            lcid, dims = s.split("=", 1)
            x, y = dims.split(":", 1)
            out[int(lcid)] = (parse_dim_name(x), parse_dim_name(y))
        except ValueError as e:
            raise SystemExit(f"--curve {s!r}: expected LCID=<xdim>:<ydim> "
                             f"({e})")
    return out


def cmd_convert(args) -> int:
    dst = parse_system(args.to)
    if args.src:
        src = parse_system(args.src)
    else:
        v = detect(args.deck, follow_includes=args.follow_includes)
        if v.system is None or v.ambiguous:
            print(v.table())
            print("\nauto-detection is not confident enough - pass --from "
                  "explicitly.", file=sys.stderr)
            return 2
        src = v.system
        print(f"auto-detected source units: {src.describe()}")
    if src == dst:
        print("source and target unit systems are identical - nothing to do.")
        return 0

    deck = Path(args.deck)
    if args.in_place and args.output:
        print("ERROR: --in-place and -o/--output are mutually exclusive.",
              file=sys.stderr)
        return 2
    if args.in_place:
        out = deck
    else:
        out = Path(args.output) if args.output else deck.with_name(
            f"{deck.stem}__{dst.key}{deck.suffix}")

    try:
        ctx = convert(str(deck), src, dst, str(out),
                      blast_unit=5 if args.blast_unit5 else None,
                      allow_unknown=args.allow_unknown,
                      follow_includes=args.follow_includes,
                      dry_run=args.dry_run,
                      curve_overrides=parse_curve_overrides(args.curve),
                      self_check=not args.no_self_check,
                      verify_roundtrip=args.verify_roundtrip,
                      backup=not args.no_backup)
    except ConvertError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    txt = report(ctx, src, dst)
    if args.dry_run:
        print("DRY RUN - no files written\n")
    print(txt)
    if not args.dry_run and not args.no_log:
        log = str(out) + ".kunit.log"
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"kunit conversion log for {deck}\n\n" + txt + "\n")
        print(f"\nlog       : {log}")
    if ctx.self_check and ctx.self_check.startswith("FAILED"):
        return 3
    return 0


def cmd_gui(_args) -> int:
    from .gui import main as gui_main
    return gui_main()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kunit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version",
                    version=f"%(prog)s {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="auto-detect a deck's unit system")
    p.add_argument("deck")
    p.add_argument("--no-includes", action="store_true",
                   help="do not follow *INCLUDE files for evidence")
    p.add_argument("--json", action="store_true",
                   help="machine-readable JSON verdict on stdout")
    p.set_defaults(fn=cmd_detect)

    p = sub.add_parser("check", help="report keyword coverage / convertibility "
                                     "without converting")
    p.add_argument("deck")
    p.add_argument("--follow-includes", action="store_true",
                   help="check the whole *INCLUDE tree")
    p.add_argument("--json", action="store_true",
                   help="machine-readable JSON report on stdout")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("convert", help="convert a deck between unit systems")
    p.add_argument("deck")
    p.add_argument("--to", required=True, help="target units, e.g. ton-mm-s")
    p.add_argument("--from", dest="src", default=None,
                   help="source units (default: auto-detect)")
    p.add_argument("-o", "--output", default=None,
                   help="output path (default: <deck>__<to>.k)")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite the input (keeps a .orig_<from> backup)")
    p.add_argument("--no-backup", action="store_true",
                   help="with --in-place: skip the backup copies")
    p.add_argument("--follow-includes", action="store_true",
                   help="convert the whole *INCLUDE tree (children are "
                        "written next to their sources; references rewritten)")
    p.add_argument("--dry-run", action="store_true",
                   help="analyse and report only, write nothing")
    p.add_argument("--curve", action="append", metavar="LCID=X:Y",
                   help="declare a curve's axis dimensions, e.g. "
                        "--curve 17=time:accel (repeatable; overrides "
                        "whatever the referencing keywords imply)")
    p.add_argument("--blast-unit5", action="store_true",
                   help="force *LOAD_BLAST[_ENHANCED] UNIT=5 + CFM/CFL/CFT/CFP "
                        "even when a built-in UNIT exists for the target")
    p.add_argument("--allow-unknown", action="store_true",
                   help="convert even if unknown keywords are present "
                        "(they are left unchanged - review them!)")
    p.add_argument("--no-self-check", action="store_true",
                   help="skip re-detecting the output as a sanity check")
    p.add_argument("--verify-roundtrip", action="store_true",
                   help="convert the output back and forward again and prove "
                        "the result is reproduced exactly")
    p.add_argument("--no-log", action="store_true",
                   help="do not write the <out>.kunit.log report file")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("systems", help="list preset unit systems")
    p.set_defaults(fn=cmd_systems)

    p = sub.add_parser("gui", help="open the graphical interface")
    p.set_defaults(fn=cmd_gui)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
