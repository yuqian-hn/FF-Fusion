# Project Structure

This document describes the current workspace structure and the parts that are most relevant for a public GitHub release.

## Current Workspace Layout

```text
FF-Fusion/
|- code/
|  |- code/                         # training-oriented fusion code
|  `- 部署用，精简版本/
|     `- code-V_deplpy/            # deployment-oriented compact code
|- result/
|  |- benchmark_results(1).json    # edge benchmark summary
|  `- 定量指标统计(1).xlsx          # quantitative fusion metrics
|- FF-Fusion-architecture.svg      # architecture figure
|- FF-Fusion-architecture.mmd      # editable mermaid source
|- FF-Fusion A Teacher-Student Distillation Framework ...
|- 融合部分实验方案.docx
|- 个人语境ff-fusion.md
|- 场景角色.md
`- github-release-docs/            # this documentation bundle
   |- source-code/
   |  |- training/                 # copied training source snapshot
   |  `- deployment/               # copied deployment source snapshot
   `- ...
```

## Code Roles

### `code/code`

Primary training-oriented implementation.

Typical entry files:

- `train.py`
- `train_student.py`
- `test1.py`
- `dataloder.py`
- `models/`

### `code/部署用，精简版本/code-V_deplpy`

Compact deployment-oriented implementation and local inference scripts.

Typical contents:

- `test1.py`
- `test1_umf.py`
- `test_piafusion.py`
- `dataloder_test.py`
- `models/`
- `runs/`
- `test_img/`

## Bundled Upload Copy

This documentation bundle also contains a code snapshot intended for direct GitHub upload:

- `github-release-docs/source-code/training`
- `github-release-docs/source-code/deployment`

These copies include source files and model-definition folders, but intentionally exclude:

- checkpoints,
- runtime result folders,
- sample output folders, and
- cache folders.

## Result Sources

### `result/benchmark_results(1).json`

Used as the primary evidence source for:

- inference-only latency,
- end-to-end latency,
- FPS,
- tested image-pair count, and
- memory statistics during deployment benchmarking.

### `result/定量指标统计(1).xlsx`

Used as the primary evidence source for:

- image-fusion quality metrics,
- FP32 comparison latency values, and
- FP32 model-size comparisons used in manuscript tables.

## Suggested Public Repository Layout

If you later reorganize the repository before publishing, this is a cleaner public layout:

```text
FF-Fusion/
|- README.md
|- LICENSE
|- docs/
|- assets/
|- training/
|- deployment/
|- results/
|- checkpoints/
`- scripts/
```

Recommended mapping:

- `training/` <- current `code/code`
- `deployment/` <- current `code/部署用，精简版本/code-V_deplpy`
- `results/` <- selected files from current `result/`
- `assets/` <- architecture figure and sample visuals
- `docs/` <- this folder

## Release Advice

- Keep manuscript files out of the public repository unless you explicitly want to publish the writing history.
- Remove temporary Word lock files such as `~$...docx`.
- Decide whether to release pretrained weights directly or provide external download links.
