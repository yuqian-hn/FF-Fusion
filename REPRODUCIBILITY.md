# Reproducibility Notes

This project already contains usable code and result files, but several items should be clarified before a public GitHub release.

## Main Entry Points

Training-oriented code:

- `code/code/train.py`
- `code/code/train_student.py`
- `code/code/test1.py`

Deployment-oriented code:

- `code/部署用，精简版本/code-V_deplpy/test1.py`
- `code/部署用，精简版本/code-V_deplpy/test1_umf.py`
- `code/部署用，精简版本/code-V_deplpy/test_piafusion.py`

Bundled upload copy:

- `github-release-docs/source-code/training`
- `github-release-docs/source-code/deployment`

## Known Path Issues

The current training scripts still contain hardcoded dataset paths pointing to a local Linux machine. These should be parameterized before public release.

Files that currently need path cleanup:

- `code/code/dataloder.py`
- `code/code/dataloder_test.py`
- `code/code/train.py`
- `code/code/train_student.py`

## Split and Protocol Notes

Current state of the fusion code:

- the fusion training scripts read the dataset directory directly,
- an explicit train-validation-test split is not exposed in the released fusion scripts,
- test scripts also use direct folder-based loading.

For a clean public release, publish:

- explicit split file lists, or
- a script that generates the split deterministically.

## Recommended Minimum Release Additions

- `requirements.txt` or `environment.yml`
- dataset-path configuration through CLI arguments or a config file
- clear instructions for training the teacher and student models
- exact checkpoint names used in the manuscript
- public sample inputs and outputs

## Suggested Public Reproduction Steps

1. Prepare the visible and infrared image folders.
2. Apply the documented preprocessing pipeline if raw data are released.
3. Configure dataset paths instead of editing source files manually.
4. Train the teacher model.
5. Train the student model with feature distillation.
6. Run the deployment-oriented inference script on the compact student checkpoint.
7. Verify the output images and compare the runtime against the published benchmark JSON.

## Recommended Terminology Alignment

For consistency in a public repository:

- use `CMF` for the deep fusion block instead of `Cross-Trans`
- use `DBB` for the student feature modeling block
- describe runtime in `ms/image` for FP32 comparison tables
- report deployment runtime as `ms` and `FPS`
