#!/usr/bin/env python
"""Thin wrapper so `python kunit.py ...` keeps working; the real CLI lives in
kunit/cli.py (installed as the `kunit` console command via pyproject.toml)."""
import sys

from kunit.cli import main

if __name__ == "__main__":
    sys.exit(main())
