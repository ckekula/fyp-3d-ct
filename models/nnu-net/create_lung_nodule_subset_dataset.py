#!/usr/bin/env python3
"""Create a smaller nnU-Net raw dataset from the lung nodule manifest.

The subset hardlinks CT images from their original locations by default and
creates binary labels from the RexGrounding segmentation channels. Cases are
selected by largest nodule mask pixel count from the manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json


DEFAULT_MANIFEST = Path("/home/chest_ct/code/data/rexgrounding-ct/lung_nodule_ct_scans.json")
DEFAULT_RAW = Path("/home/chest_ct/code/models/nnu-net/storage/nnUNet_raw")


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def nodule_pixels(entry: dict) -> int:
    pixels = entry.get("pixels", {})
    return sum(int(value or 0) for value in pixels.values())


def nodule_channels(entry: dict) -> list[int]:
    return sorted(int(channel) for channel in entry.get("nodule_channels", []))


def collapse_channels(seg_path: Path, channels: list[int]) -> np.ndarray:
    seg_img = nib.load(str(seg_path))
    data = seg_img.dataobj

    if len(seg_img.shape) == 3:
        if channels != [0]:
            raise ValueError(f"{seg_path} is 3D but requested channels {channels}")
        return (np.asanyarray(data) > 0).astype(np.uint8)

    if len(seg_img.shape) != 4:
        raise ValueError(f"Expected 3D or 4D segmentation, got {seg_img.shape}: {seg_path}")

    mask = np.zeros(seg_img.shape[1:], dtype=bool)
    for channel in channels:
        if channel >= seg_img.shape[0]:
            raise ValueError(
                f"Channel {channel} out of range for {seg_path} with shape {seg_img.shape}"
            )
        mask |= np.asanyarray(data[channel]) > 0
    return mask.astype(np.uint8)


def write_label(entry: dict, out_path: Path) -> None:
    ct_path = Path(entry["ct_path"])
    seg_path = Path(entry["seg_path"])
    mask = collapse_channels(seg_path, nodule_channels(entry))

    ct_img = nib.load(str(ct_path))
    if mask.shape != ct_img.shape:
        raise ValueError(
            f"Shape mismatch for {entry['case_id']}: mask {mask.shape}, CT {ct_img.shape}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    label_img = nib.Nifti1Image(mask, ct_img.affine, ct_img.header.copy())
    label_img.set_data_dtype(np.uint8)
    nib.save(label_img, str(out_path))


def convert_entry(entry: dict, images_dir: Path, labels_dir: Path, link_mode: str) -> bool:
    case_id = entry["case_id"]
    ct_path = Path(entry["ct_path"])
    seg_path = Path(entry["seg_path"])
    if not ct_path.exists() or not seg_path.exists():
        return False

    link_or_copy(ct_path, images_dir / f"{case_id}_0000.nii.gz", link_mode)
    label_out = labels_dir / f"{case_id}.nii.gz"
    if not label_out.exists():
        write_label(entry, label_out)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--nnunet-raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--dataset-id", type=int, default=103)
    parser.add_argument("--dataset-name", default="LungNodules2DSubset120")
    parser.add_argument("--num-train", type=int, default=120)
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_full_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    target_dir = args.nnunet_raw / dataset_full_name

    with args.manifest.open() as f:
        manifest = json.load(f)

    train_entries = sorted(
        manifest["splits"]["train"],
        key=lambda entry: (nodule_pixels(entry), entry["case_id"]),
        reverse=True,
    )[: args.num_train]
    test_entries = manifest["splits"].get("test", [])

    converted_train = 0
    for entry in train_entries:
        if convert_entry(entry, target_dir / "imagesTr", target_dir / "labelsTr", args.link_mode):
            converted_train += 1

    copied_test = 0
    for entry in test_entries:
        if convert_entry(entry, target_dir / "imagesTs", target_dir / "labelsTs", args.link_mode):
            copied_test += 1

    generate_dataset_json(
        output_folder=target_dir,
        channel_names={0: "CT"},
        labels={"background": 0, "lung_nodule": 1},
        num_training_cases=converted_train,
        file_ending=".nii.gz",
        dataset_name=dataset_full_name,
        reference="RexGrounding-CT",
        description=(
            f"Top {converted_train} largest-mask lung nodule cases from "
            "RexGrounding-CT."
        ),
    )

    print("Created subset dataset")
    print(f"  dataset : {dataset_full_name}")
    print(f"  path    : {target_dir}")
    print(f"  train   : {converted_train}")
    print(f"  test    : {copied_test}")
    print(f"  link    : {args.link_mode}")
    print()
    print("Next command:")
    print(
        f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
        "--verify_dataset_integrity -c 2d -np 2 --no_pbar"
    )


if __name__ == "__main__":
    main()
