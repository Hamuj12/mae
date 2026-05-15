# ODAD Threshold Policy Summary

**Recommended report point:** Memory-bank ODAD at static confidence threshold 0.15.

At conf=0.15, memory-bank ODAD reaches 88.4% detection with 6.2% weird-box rate, 3.0% box-jump rate, 16.0% bad-frame proxy rate, and max bad streak 51.

Interpretation: memory-bank ODAD is mainly under-confident at the default 0.25 threshold. Static threshold calibration recovers detection while preserving substantially better proxy reliability than current ODAD. Adaptive policies were cleaner by proxy on low-confidence accepts, but did not beat static 0.15 on the overall detection/reliability tradeoff. Static 0.05 remains an aggressive/manual-review candidate.

| Model | Conf. | Det. | Weird | Jump | Bad | Max Bad | Note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FDA-mix | 0.25 | 98.2% | 21.5% | 1.9% | 23.5% | 41 | High detection baseline; reliability proxy failures remain elevated. |
| Current ODAD | 0.25 | 87.0% | 36.3% | 9.1% | 45.9% | 198 | Existing ODAD baseline at default threshold. |
| Adapter Base | 0.25 | 84.5% | 6.3% | 3.1% | 19.5% | 71 | Adapter baseline from reliability_adapter_l18_l21_full full-stream summary. |
| Memory-bank ODAD | 0.25 | 85.6% | 6.2% | 3.0% | 18.2% | 67 | Default threshold; under-confident for memory-bank ODAD. |
| Memory-bank ODAD | 0.20 | 86.9% | 6.2% | 3.0% | 17.2% | 67 | Intermediate threshold in static sweep. |
| Memory-bank ODAD | 0.15 | 88.4% | 6.2% | 3.0% | 16.0% | 51 | Recommended reportable/deployable operating point. |
| Memory-bank ODAD | 0.10 | 90.3% | 6.2% | 3.0% | 14.5% | 49 | Higher-recall static point; not selected as main report point. |
| Memory-bank ODAD | 0.05 | 93.0% | 6.2% | 3.0% | 12.3% | 46 | Aggressive/manual-review candidate only. |

Proxy metrics are not label-aware accuracy. They summarize detection continuity, weird boxes, box jumps, and bad streaks over the ordered lab-image stream.
