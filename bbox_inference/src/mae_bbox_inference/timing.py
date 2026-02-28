"""Timing helpers for bbox inference"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence
import numpy as np


@dataclass
class TimingSummary:
    """Aggregate latency statistics in milliseconds"""

    # number of finite latency samples used in summary
    count: int
    # central tendency and tail stats for runtime diagnostics
    mean_ms: float
    min_ms: float
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float


def time_call(fcn: Callable, *args, **kwargs):
    """Time a callable in milliseconds and return (result, elapsed_ms)"""

    # lightweight timer wrapper reused by inference and warmup
    t0          = perf_counter()
    result      = fcn(*args, **kwargs)
    elapsed_ms  = (perf_counter() - t0) * 1000.0
    return result, float(elapsed_ms)


def summarize_latencies(latencies_ms: Sequence[float]) -> TimingSummary:
    """Compute summary statistics for latency samples"""

    # flatten to 1-D and drop NaN/Inf values from failed runs
    vals = np.asarray(latencies_ms, dtype = float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        # return explicit NaN stats so downstream logging stays consistent
        return TimingSummary(
                                count = 0,
                                mean_ms = float('nan'),
                                min_ms = float('nan'),
                                p50_ms = float('nan'),
                                p90_ms = float('nan'),
                                p99_ms = float('nan'),
                                max_ms = float('nan'),
                            )

    # percentile set matches common deployment latency reporting
    return TimingSummary(
                            count = int(vals.size),
                            mean_ms = float(np.mean(vals)),
                            min_ms = float(np.min(vals)),
                            p50_ms = float(np.percentile(vals, 50)),
                            p90_ms = float(np.percentile(vals, 90)),
                            p99_ms = float(np.percentile(vals, 99)),
                            max_ms = float(np.max(vals)),
                        )
