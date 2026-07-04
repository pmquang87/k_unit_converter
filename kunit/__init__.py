"""kunit - LS-DYNA .k deck unit-system converter with auto-detection."""
from .convert import ConvertError, convert, report, scan
from .detect import detect
from .units import PRESETS, UnitSystem, factor, parse_system

__version__ = "0.1.0"
__all__ = ["convert", "report", "scan", "detect", "ConvertError",
           "UnitSystem", "PRESETS", "parse_system", "factor"]
