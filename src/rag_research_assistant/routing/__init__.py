"""Interpretable routing primitives for isolated retrieval experiments."""

from .signals import (
    BaselineSignals,
    RankedCandidate,
    RoutingMetrics,
    ThresholdRule,
    evaluate_routing_decisions,
    extract_baseline_signals,
)

__all__ = [
    "BaselineSignals",
    "RankedCandidate",
    "RoutingMetrics",
    "ThresholdRule",
    "evaluate_routing_decisions",
    "extract_baseline_signals",
]
