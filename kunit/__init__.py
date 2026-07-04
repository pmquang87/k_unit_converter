"""kunit - LS-DYNA .k deck unit-system converter with auto-detection."""
from .convert import ConvertError, convert, load_tree, report, scan
from .detect import detect
from .units import PRESETS, UnitSystem, factor, parse_dim_name, parse_system

__version__ = "0.2.0"
__all__ = ["convert", "report", "scan", "detect", "ConvertError", "load_tree",
           "UnitSystem", "PRESETS", "parse_system", "parse_dim_name", "factor"]
