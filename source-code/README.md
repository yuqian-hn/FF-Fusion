# Source Code Bundle

This folder contains a cleaned source-code snapshot prepared for direct upload together with the documentation bundle.

## Included Subfolders

- `training/`: training-oriented fusion code copied from `code/code`
- `deployment/`: deployment-oriented compact code copied from `code/部署用，精简版本/code-V_deplpy`

## What Was Kept

- Python source files
- model-definition folders
- inference entry scripts
- training entry scripts

## What Was Intentionally Excluded

- checkpoints and weight files
- local result folders
- sample output images
- compressed archives
- cache folders such as `__pycache__`

## Notes

- Some scripts still contain hardcoded dataset paths and should be cleaned before final public release.
- If you want a fully runnable public repository, add environment setup files and configurable dataset paths.
