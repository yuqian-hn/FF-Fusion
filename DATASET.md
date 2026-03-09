# TF-1770 Dataset Notes

## Overview

`TF-1770 (Forest Fire Fusion-1770)` is a visible-infrared image dataset constructed for real forest fire perception scenarios.

Verified high-level properties:

- `1,770` synchronized visible-infrared image pairs
- Two acquisition viewpoints:
  - firefighter first-person observation
  - UAV aerial observation
- Scene elements include:
  - fire
  - smoke
  - firefighting personnel
- Challenging conditions include:
  - smoke occlusion
  - low illumination
  - multi-scale flame targets

## Pair Construction Pipeline

The image-pair construction logic used in the manuscript is:

1. Synchronously collect visible and thermal infrared observations.
2. Perform pixel-level registration using the enhanced correlation coefficient (ECC) method.
3. Because the visible sensor has a larger field of view than the thermal sensor, crop the visible image to the effective overlapping region after registration.
4. Resize the aligned pair to a unified resolution of `640 x 480`.

This description is suitable for public documentation because it explains why the visible image needs both alignment and cropping before fusion.

## Annotation Notes

The project text states that polygon-based annotations are prepared for downstream tasks such as:

- object detection
- image segmentation

If these annotations are released publicly, publish them together with:

- category definitions,
- file naming rules,
- split lists, and
- annotation examples.

## Split Notes

Important reproducibility note:

- The current fusion training scripts in the workspace do not yet expose an explicit train-validation-test split.
- If the public repository reports a split-based protocol for downstream tasks, publish the exact split file lists together with the annotations.

## Recommended Public Dataset Package

When the dataset is released, the package should ideally include:

- visible images
- infrared images
- polygon annotations
- category list
- split files
- dataset license
- Zenodo DOI
- a short usage guide

## Suggested Dataset Description

`TF-1770 is a forest-fire-oriented visible-infrared dataset containing 1,770 synchronized image pairs collected from firefighter and UAV viewpoints. Each pair is spatially aligned by ECC-based registration, cropped to the overlapping field of view, and resized to 640 x 480 for multimodal fusion and deployment-oriented evaluation.`
