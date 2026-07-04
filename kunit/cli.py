"""kunit command line interface.

  kunit detect  deck.k
  kunit convert deck.k --to ton-mm-s [--from kg-m-s] [-o out.k]
  kunit convert deck.k --to ton-mm-s --in-place --follow-includes
  kunit systems
  kunit gui
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .convert import ConvertError, convert, report
from .detect import detect
from .units import PRESETS, parse_dim_name, parse_system


def cmd_systems(_args) -> int:
    print("preset unit systems (custom MASS-LENGTH-TIME triples also accepted):")
    for s in PRESETS.values():
        print(f"  {s.describe()}")
    return 0


def cmd_detect(args) -> int:
    v = detect(args.deck, follow_includes=not args.no_includes)
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
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="auto-detect a deck's unit system")
    p.add_argument("deck")
    p.add_argument("--no-includes", action="store_true",
                   help="do not follow *INCLUDE files for evidence")
    p.set_defaults(fn=cmd_detect)

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
