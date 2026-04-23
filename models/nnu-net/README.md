# nnU-Net

# Table of Contents

- [Installation and Setup](#installation-and-setup)
  - [1. Check the basics](#1-check-the-basics)
  - [2. Install PyTorch first](#2-install-pytorch-first)
  - [3. Install nnU-Net](#3-install-nnu-net)
  - [4. Create the three storage locations](#4-create-the-three-storage-locations)
  - [5. Set environment variables](#5-set-environment-variables)
  - [6. Verify the setup](#6-verify-the-setup)
  - [7. Optional extras](#7-optional-extras)
- [Prepare a Dataset](#prepare-a-dataset)
  - [Required format](#required-format)
  - [Input formats](#input-formats)
  - [Inference inputs](#inference-inputs)
- [Plan and Preprocess](#plan-and-preprocess)
  - [Recommended command](#recommended-command)
  - [What this does](#what-this-does)
  - [Useful options](#useful-options)
  - [Split commands](#split-commands)
  - [What to inspect afterward](#what-to-inspect-afterward)
- [Train Models](#train-models)
  - [Training overview](#training-overview)
  - [Basic training command](#basic-training-command)
  - [Important flag for later model selection](#important-flag-for-later-model-selection)
  - [Device selection](#device-selection)
  - [Recommended multi-GPU usage](#recommended-multi-gpu-usage)
  - [Output location](#output-location)
  - [Next steps](#next-steps)
- [Run Inference](#run-inference)
  - [Before you start](#before-you-start)
  - [Predict with a trained configuration](#predict-with-a-trained-configuration)
  - [Ensemble multiple configuration outputs](#ensemble-multiple-configuration-outputs)
  - [Apply postprocessing](#apply-postprocessing)
  - [Predict from a model folder](#predict-from-a-model-folder)
  - [Export and import your own trained model](#export-and-import-your-own-trained-model)
  - [Public pretrained models](#public-pretrained-models)

# Installation and Setup

This guide consolidates the setup steps needed for a first nnU-Net v2 run.

## 1. Check the basics

- Use Python 3.10 or newer.
- Linux is the primary target, but Windows and macOS are also supported.
- GPU is strongly recommended for training.

## 2. Install PyTorch first

Install PyTorch for your hardware before installing `nnunetv2`:

<https://pytorch.org/get-started/locally/>

Choose the build that matches your environment:

- `cuda` for NVIDIA GPUs
- `cpu` if no accelerator is available

Do not install `nnunetv2` before PyTorch is in place.

## 3. Install nnU-Net

For normal use:

```bash
pip install nnunetv2
```

If you want a local editable checkout for development:

```bash
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

## 4. Create the three storage locations

nnU-Net needs three locations:

- `nnUNet_raw`: raw datasets in nnU-Net format
- `nnUNet_preprocessed`: preprocessed data used during training
- `nnUNet_results`: trained models and installed pretrained models

Recommended layout:

```text
/path/to/nnUNet_raw
/path/to/nnUNet_preprocessed
/path/to/nnUNet_results
```

## 5. Set environment variables

### Linux and macOS

For a persistent setup, add this to your shell profile such as `.bashrc` or `.zshrc`:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

For a temporary setup, run the same commands in the current shell before using nnU-Net.

## 6. Verify the setup

Check that the variables are visible in your shell.

Linux and macOS:

```bash
echo "$nnUNet_raw"
echo "$nnUNet_preprocessed"
echo "$nnUNet_results"
```

## 7. Optional extras

`hiddenlayer` enables network topology plots:

```bash
pip install --upgrade git+https://github.com/FabianIsensee/hiddenlayer.git
```

If you train on a fast GPU, you may also want to tune `nnUNet_n_proc_DA` for data augmentation throughput.

# Prepare a Dataset

## Required format

nnU-Net expects datasets in the nnU-Net dataset format. Start with the concise reference here:

- [Dataset and input format reference](../reference/dataset-format.md)

The key points are:

- each dataset lives in `nnUNet_raw/DatasetXXX_Name`
- training images go into `imagesTr`
- training labels go into `labelsTr`
- optional test images go into `imagesTs`
- `dataset.json` describes modalities and labels

## Input formats

nnU-Net v2 supports multiple file formats. The exact supported formats and image I/O details are documented in:

- [nnU-Net dataset format](../dataset_format.md#supported-file-formats)

## Inference inputs

Inference input folders follow the training dataset's naming and file-ending conventions:

- NaturalImage2DIO: .png, .bmp, .tif
- NibabelIO: .nii.gz, .nrrd, .mha
- NibabelIOWithReorient: .nii.gz, .nrrd, .mha. This reader will reorient images to RAS!
- SimpleITKIO: .nii.gz, .nrrd, .mha
- Tiff3DIO: .tif, .tiff. 3D tif images! Since TIF does not have a standardized way of storing spacing information, nnU-Net expects each TIF file to be accompanied by an identically named .json file that contains this information

# Plan and Preprocess

This guide covers dataset fingerprint extraction, experiment planning, and preprocessing.

## Recommended command

For a new dataset, use:

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

`DATASET_ID` is the numeric dataset identifier. `--verify_dataset_integrity` is recommended the first time you run the command.

## What this does

The command performs three steps:

1. Extract a dataset fingerprint
2. Create one or more nnU-Net configurations
3. Preprocess the data for those configurations

The output is written into `nnUNet_preprocessed/DatasetXXX_Name`.

## Useful options

- Use `--no_pbar` in non-interactive environments.
- Use `-d 1 2 3` to process multiple datasets.
- Use `-c 3d_fullres` if you already know which configuration you want.
- Use `-h` to inspect all options.

## Split commands

If you need more control, you can run the steps individually:

```bash
nnUNetv2_extract_fingerprint -d DATASET_ID
nnUNetv2_plan_experiment -d DATASET_ID
nnUNetv2_preprocess -d DATASET_ID
```

## What to inspect afterward

After preprocessing, the dataset folder in `nnUNet_preprocessed` contains:

- `dataset_fingerprint.json`
- `nnUNetPlans.json`
- preprocessed data folders for the created configurations

# Train Models

## Training overview

nnU-Net can create several configurations depending on the dataset:

- `2d`
- `3d_fullres`
- `3d_lowres`
- `3d_cascade_fullres`

Not every dataset gets every configuration. Small datasets may not create the cascade.

Training is usually done as a 5-fold cross-validation so nnU-Net can compare configurations and optionally ensemble them later.

## Basic training command

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD
```

Examples:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_lowres 0
nnUNetv2_train DATASET_NAME_OR_ID 3d_cascade_fullres 0
```

For the cascade, `3d_lowres` must be trained before `3d_cascade_fullres`.

## Important flag for later model selection

If you plan to use `nnUNetv2_find_best_configuration`, train with `--npz`:

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD --npz
```

This stores validation probabilities needed for automatic configuration comparison and ensembling.

If you already trained without `--npz`, you can rerun validation:

```bash
nnUNetv2_train DATASET_NAME_OR_ID CONFIGURATION FOLD --val --npz
```

## Device selection

Use `-device` to choose `cpu`, `cuda`, or `mps`.

For multi-GPU systems, select the GPU with `CUDA_VISIBLE_DEVICES`:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 3d_fullres 0 --npz
```

## Recommended multi-GPU usage

If you have multiple GPUs, the preferred strategy is usually one training per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train DATASET_NAME_OR_ID 2d 1 --npz
```

Distributed training is also available:

```bash
nnUNetv2_train DATASET_NAME_OR_ID 2d 0 --npz -num_gpus X
```

## Output location

Training outputs are written under:

```text
nnUNet_results/DatasetXXX_Name/TRAINER__PLANS__CONFIGURATION/fold_X
```

Important artifacts include:

- `checkpoint_final.pth`
- `checkpoint_best.pth`
- `progress.png`
- `validation/summary.json`
- `validation/*.npz` if `--npz` was enabled

## Next steps

- [Find the best configuration](find-best-configuration.md)

# Run Inference

## Before you start

Input images must match the trained dataset's naming convention and file endings. See:

- [Dataset and input format reference](../reference/dataset-format.md)

If you previously ran `nnUNetv2_find_best_configuration`, use the commands it generated in `inference_instructions.txt` whenever possible.

## Predict with a trained configuration

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION
```

If you want to ensemble probability outputs from multiple configurations, add `--save_probabilities`:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
```

By default, inference uses the 5 trained folds as an ensemble. If you trained the `all` fold and want to use only that model:

```bash
nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION -f all
```

## Ensemble multiple configuration outputs

```bash
nnUNetv2_ensemble -i FOLDER1 FOLDER2 -o OUTPUT_FOLDER -np NUM_PROCESSES
```

The input folders must contain probability files produced with `--save_probabilities`.

## Apply postprocessing

```bash
nnUNetv2_apply_postprocessing \
  -i FOLDER_WITH_PREDICTIONS \
  -o OUTPUT_FOLDER \
  --pp_pkl_file POSTPROCESSING_FILE \
  -plans_json PLANS_FILE \
  -dataset_json DATASET_JSON_FILE
```

For single-configuration predictions, `plans.json` and `dataset.json` are usually copied automatically. For ensemble outputs, provide them explicitly.

## Predict from a model folder

If you want to run inference directly from an exported or copied model folder:

```bash
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

## Export and import your own trained model

To move a trained model to another machine:

1. Export it:

```bash
nnUNetv2_export_model_to_zip -d DATASET_NAME_OR_ID -o MODEL.zip
```

2. Install it on the target machine:

```bash
nnUNetv2_install_pretrained_model_from_zip MODEL.zip
```

The target machine still needs a compatible nnU-Net installation and all dependencies.

## Public pretrained models

The old page on pretrained-model inference remains here:

- [How to run inference with pretrained models](../run_inference_with_pretrained_models.md)

Check that page for the current status before relying on it.