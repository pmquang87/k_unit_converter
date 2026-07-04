#!/usr/bin/env pythonw
"""Double-clickable launcher for the kunit GUI.

The .pyw extension makes Windows run it with pythonw.exe, so no console
window opens behind the GUI. `python run_gui.pyw` works from a terminal too.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kunit.gui import main

if __name__ == "__main__":
    sys.exit(main())
