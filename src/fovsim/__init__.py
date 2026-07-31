"""FoV progressive-coding simulation infrastructure."""

from .policy import PolicyConfig, classify_fraction
from .trace import TraceRow, TraceSummary, load_trace

__all__ = [
    "PolicyConfig",
    "TraceRow",
    "TraceSummary",
    "classify_fraction",
    "load_trace",
]

__version__ = "0.1.0"
