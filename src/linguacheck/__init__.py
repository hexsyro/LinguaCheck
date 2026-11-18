"""LinguaCheck — catch sentiment scored with the wrong language model."""

from linguacheck.common import RESULT_SCHEMA_VERSION, ToolResult
from linguacheck.logic import LinguaCheckLogic, run

__version__ = "0.2.0"
__all__ = [
    "RESULT_SCHEMA_VERSION",
    "LinguaCheckLogic",
    "ToolResult",
    "__version__",
    "run",
]
