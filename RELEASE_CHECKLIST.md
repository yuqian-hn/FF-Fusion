# Public Release Checklist

Use this checklist before uploading the project to GitHub.

## Repository Basics

- Add `README.md` at the repository root.
- Add a license file.
- Add contact information or a maintainer email.
- Add a citation entry or BibTeX snippet.

## Code Cleanup

- Replace hardcoded dataset paths with configurable arguments.
- Remove temporary files and caches.
- Remove unnecessary local result folders if they are only test artifacts.
- Verify that script names and module names are consistent with the manuscript terminology.

## Dataset Release

- Confirm that the dataset can be redistributed publicly.
- Publish the Zenodo link or DOI.
- Include annotation files if downstream tasks are claimed.
- Include class definitions and split files.

## Model Release

- Decide which checkpoints should be public.
- Verify that model names in the repository match the manuscript.
- If INT8 deployment artifacts are claimed, publish the compiled artifacts or remove the size claim.

## Documentation

- Keep the architecture figure in the repository.
- Document the training workflow and the deployment workflow separately.
- Explain the preprocessing pipeline for visible-infrared alignment.
- Explain the difference between FP32 model size and deployment artifact size if both are reported.

## Results

- Keep the benchmark JSON and quantitative spreadsheet, or export them into cleaner CSV or Markdown tables.
- Make sure all reported latency units are consistent.
- If the same latency appears across multiple datasets, explain that it is method-level runtime under a fixed setting.

## Final Verification

- Clone the repository into a clean location.
- Run at least one training or inference command from the public instructions.
- Verify that all links in the README work.
- Confirm that no private paths, personal information, or non-redistributable files remain.
