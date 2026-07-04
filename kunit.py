#!/usr/bin/env python
"""kunit - convert an LS-DYNA .k deck between unit systems.

  python kunit.py detect  deck.k
  python kunit.py convert deck.k --to ton-mm-s [--from kg-m-s] [-o out.k]
  python kunit.py convert deck.k --to ton-mm-s --in-place
  python kunit.py systems
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from kunit import (ConvertError, PRESETS, convert, detect, parse_system,
                   report)


def cmd_systems(_args) -> int:
    print("preset unit systems (custom MASS-LENGTH-TIME triples also accepted):")
    for s in PRESETS.values():
        print(f"  {s.describe()}")
    return 0


def cmd_detect(args) -> int:
    v = detect(args.deck)
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


def cmd_convert(args) -> int:
    dst = parse_system(args.to)
    if args.src:
        src = parse_system(args.src)
    else:
        v = detect(args.deck)
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
        if not args.no_backup:
            bak = deck.with_suffix(deck.suffix + f".orig_{src.key}")
            if bak.exists():
                print(f"backup {bak} already exists - refusing to overwrite it.",
                      file=sys.stderr)
                return 2
            shutil.copy2(deck, bak)
            print(f"backup    : {bak}")
    else:
        out = Path(args.output) if args.output else deck.with_name(
            f"{deck.stem}__{dst.key}{deck.suffix}")

    try:
        ctx = convert(str(deck), src, dst, str(out),
                      blast_unit=5 if args.blast_unit5 else None,
                      allow_unknown=args.allow_unknown)
    except ConvertError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"written   : {out}\n")
    print(report(ctx, src, dst))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="kunit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("detect", help="auto-detect a deck's unit system")
    p.add_argument("deck")
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
                   help="with --in-place: skip the backup copy")
    p.add_argument("--blast-unit5", action="store_true",
                   help="force *LOAD_BLAST[_ENHANCED] UNIT=5 + CFM/CFL/CFT/CFP "
                        "even when a built-in UNIT exists for the target")
    p.add_argument("--allow-unknown", action="store_true",
                   help="convert even if unknown keywords are present "
                        "(they are left unchanged - review them!)")
    p.set_defaults(fn=cmd_convert)

    p = sub.add_parser("systems", help="list preset unit systems")
    p.set_defaults(fn=cmd_systems)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
