# Result Summary

This file summarizes the main verified results that are already supported by files in the current workspace.

## Evidence Sources

- Quantitative fusion metrics: `result/定量指标统计(1).xlsx`
- Deployment benchmark: `result/benchmark_results(1).json`

## Edge Deployment Benchmark

The deployment benchmark JSON reports evaluation on `1770` visible-infrared image pairs with input size `640 x 480`.

| Item | Value |
| --- | ---: |
| Inference-only latency | 14.19 ms |
| Inference-only FPS | 70.48 |
| End-to-end latency | 36.97 ms |
| End-to-end FPS | 27.05 |
| Average preprocessing time | 22.78 ms |
| Load time | 103.54 ms |

## TF-1770 Quantitative Summary for FF-Fusion

The following values are taken from the verified quantitative record used in the manuscript tables.

| Metric | Value |
| --- | ---: |
| SD | 43.66 |
| MI | 2.49 |
| VIF | 0.65 |
| AG | 10.83 |
| EN | 7.10 |
| Qabf | 0.58 |
| SF | 31.08 |
| FP32 latency | 20.14 ms/image |
| FP32 model size | 1.34 MB |

## LLVIP Quantitative Summary for FF-Fusion

| Metric | Value |
| --- | ---: |
| SD | 46.38 |
| MI | 3.22 |
| VIF | 0.93 |
| AG | 4.16 |
| EN | 7.22 |
| Qabf | 0.63 |
| SF | 14.81 |
| FP32 latency | 20.14 ms/image |
| FP32 model size | 1.34 MB |

## Runtime Interpretation Note

If the same FP32 latency values are reported across multiple datasets, that is acceptable only when the runtime is measured as a method-level latency under a fixed input size, hardware environment, and evaluation setting.

Suggested note for public tables:

`Runtime is method-specific under a fixed FP32 inference setting and is therefore identical across datasets.`

## Size Interpretation Note

The currently verified compact student checkpoint in the workspace is the FP32 weight file:

- `code/部署用，精简版本/code-V_deplpy/runs/VIF_student_95.pth`

This supports the `1.34 MB` FP32 model-size statement. If you later report an INT8 deployment artifact size, publish the corresponding compiled artifact as well.
