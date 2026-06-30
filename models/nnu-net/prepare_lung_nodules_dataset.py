#!/usr/bin/env python3
"""Create a RexGrounding-CT lung nodule CT manifest.

By default this script only writes a JSON manifest of CT scans that contain
lung nodule categories. It does not copy CT volumes or generate masks.

If needed later, pass --prepare-nnunet to create an nnU-Net v2 raw dataset.
The RexGrounding segmentation files are multi-channel. Each finding is stored
in one channel and categorized in dataset.json. The optional nnU-Net converter
collapses the nodule categories into one binary label:

    background = 0
    lung_nodule = 1

By default it includes:
    2d = solitary pulmonary nodule or mass
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm.auto import tqdm


DEFAULT_REX_JSON = Path("/home/chest_ct/code/data/rexgrounding-ct/dataset.json")
DEFAULT_CT_ROOT = Path("/home/chest_ct/code/data/data_volumes/dataset/train_fixed")
DEFAULT_SEG_ROOT = Path("/home/chest_ct/code/data/segmentations/segmentations")
DEFAULT_NNUNET_RAW = Path("/home/chest_ct/code/models/nnu-net/storage/nnUNet_raw")
DEFAULT_MANIFEST = Path("/home/chest_ct/code/data/rexgrounding-ct/lung_nodule_ct_scans.json")


def resolve_ct_path(ct_root: Path, raw_name: str) -> Path:
    case_id = raw_name.removesuffix(".nii.gz")
    parts = case_id.split("_")
    train_dir = "_".join(parts[:2])
    series_dir = "_".join(parts[:-1])
    return ct_root / train_dir / series_dir / raw_name


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
            shutil.copy2(src, dst)
            return
    shutil.copy2(src, dst)


def nodule_channels(entry: dict, categories: set[str]) -> list[int]:
    return sorted(
        int(channel)
        for channel, category in entry.get("categories", {}).items()
        if category in categories
    )


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


def write_label(mask: np.ndarray, ct_path: Path, out_path: Path) -> None:
    ct_img = nib.load(str(ct_path))
    if mask.shape != ct_img.shape:
        raise ValueError(
            f"Shape mismatch for {ct_path.name}: mask {mask.shape}, CT {ct_img.shape}"
        )

    header = ct_img.header.copy()
    label_img = nib.Nifti1Image(mask, ct_img.affine, header)
    label_img.set_data_dtype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(label_img, str(out_path))


def write_dataset_json(dataset_dir: Path, dataset_name: str, num_training: int) -> None:
    dataset_json = {
        "channel_names": {"0": "CT"},
        "labels": {"background": 0, "lung_nodule": 1},
        "numTraining": num_training,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "reference": "RexGrounding-CT",
        "description": (
            "Binary lung nodule segmentation from RexGrounding-CT category 2d."
        ),
    }
    with (dataset_dir / "dataset.json").open("w") as f:
        json.dump(dataset_json, f, indent=2)
        f.write("\n")


def collect_candidates(
    entries: list[dict],
    categories: set[str],
    ct_root: Path,
    seg_root: Path,
    split: str,
    require_local_ct: bool,
) -> tuple[list[dict], Counter]:
    candidates: list[tuple[dict, list[int], Path, Path]] = []
    skipped: Counter = Counter()

    for entry in entries:
        channels = nodule_channels(entry, categories)
        if not channels:
            skipped["no_target_category"] += 1
            continue

        raw_name = entry["name"]
        ct_path = resolve_ct_path(ct_root, raw_name)
        seg_path = seg_root / raw_name

        ct_exists = ct_path.exists()
        seg_exists = seg_path.exists()

        if require_local_ct and not ct_exists:
            skipped["missing_ct"] += 1
            continue

        candidate = {
            "name": raw_name,
            "case_id": raw_name.removesuffix(".nii.gz"),
            "split": split,
            "source_split": split,
            "ct_path": str(ct_path),
            "ct_exists": ct_exists,
            "seg_path": str(seg_path),
            "seg_exists": seg_exists,
            "nodule_channels": channels,
            "nodule_categories": {
                channel: entry.get("categories", {}).get(channel)
                for channel in map(str, channels)
            },
            "nodule_findings": {
                channel: entry.get("findings", {}).get(channel)
                for channel in map(str, channels)
            },
            "shape": entry.get("shape"),
            "pixels": {
                channel: entry.get("pixels", {}).get(channel)
                for channel in map(str, channels)
            },
            "protocol": entry.get("protocol"),
        }
        candidates.append(candidate)

    return candidates, skipped


def convert_entries(
    candidates: list[dict],
    images_dir: Path,
    labels_dir: Path,
    link_mode: str,
    overwrite_labels: bool,
    max_cases: int | None,
) -> tuple[list[str], Counter]:
    converted: list[str] = []
    skipped: Counter = Counter()
    selected = candidates[:max_cases] if max_cases else candidates

    for entry in tqdm(selected, desc="Converting candidates"):
        raw_name = entry["name"]
        case_id = entry["case_id"]
        channels = entry["nodule_channels"]
        ct_path = Path(entry["ct_path"])
        seg_path = Path(entry["seg_path"])

        if not seg_path.exists():
            skipped["missing_segmentation"] += 1
            continue

        image_out = images_dir / f"{case_id}_0000.nii.gz"
        label_out = labels_dir / f"{case_id}.nii.gz"
        link_or_copy(ct_path, image_out, link_mode)

        if overwrite_labels or not label_out.exists():
            mask = collapse_channels(seg_path, channels)
            if mask.max() == 0:
                skipped["empty_target_mask"] += 1
                continue
            write_label(mask, ct_path, label_out)

        converted.append(case_id)

    return converted, skipped


def write_manifest(
    manifest_path: Path,
    categories: set[str],
    candidates_by_split: dict[str, list[dict]],
    skipped_by_split: dict[str, Counter],
    split_policy: str,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "description": "RexGrounding-CT CT scans containing lung nodule categories.",
        "split_policy": split_policy,
        "target_categories": sorted(categories),
        "category_meanings": {
            "1e": "diffusely distributed multiple pulmonary nodules",
            "2d": "solitary pulmonary nodule or mass",
        },
        "ct_root": str(DEFAULT_CT_ROOT),
        "seg_root": str(DEFAULT_SEG_ROOT),
        "counts": {
            split: len(entries)
            for split, entries in candidates_by_split.items()
        },
        "skipped": {
            split: dict(counter)
            for split, counter in skipped_by_split.items()
            if counter
        },
        "splits": candidates_by_split,
    }
    with manifest_path.open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rex-json", type=Path, default=DEFAULT_REX_JSON)
    parser.add_argument("--ct-root", type=Path, default=DEFAULT_CT_ROOT)
    parser.add_argument("--seg-root", type=Path, default=DEFAULT_SEG_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--nnunet-raw", type=Path, default=DEFAULT_NNUNET_RAW)
    parser.add_argument("--dataset-id", type=int, default=102)
    parser.add_argument("--dataset-name", default="LungNodules2D")
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["2d"],
        help="RexGrounding category codes to collapse into lung_nodule.",
    )
    parser.add_argument(
        "--link-mode",
        choices=["hardlink", "symlink", "copy"],
        default="hardlink",
        help="How to place CT images in imagesTr. Hardlink saves disk when possible.",
    )
    parser.add_argument("--overwrite-labels", action="store_true")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Convert only the first N available cases. Useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report available cases and skipped reasons; do not write files.",
    )
    parser.add_argument(
        "--include-missing-ct",
        action="store_true",
        help="Include nodule metadata even when the local CT file is missing.",
    )
    parser.add_argument(
        "--prepare-nnunet",
        action="store_true",
        help="Also create imagesTr/labelsTr for nnU-Net. This is slow.",
    )
    parser.add_argument(
        "--test-from-local-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Put locally available source val scans in the manifest test split. "
            "Enabled by default because source test CT files are not local."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_full_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_dir = args.nnunet_raw / dataset_full_name
    images_tr = dataset_dir / "imagesTr"
    labels_tr = dataset_dir / "labelsTr"

    with args.rex_json.open() as f:
        rex_data = json.load(f)

    categories = set(args.categories)
    candidates_by_split: dict[str, list[dict]] = {}
    skipped_by_split: dict[str, Counter] = {}
    for split in ("train", "val", "test"):
        candidates, skipped = collect_candidates(
            entries=rex_data.get(split, []),
            categories=categories,
            ct_root=args.ct_root,
            seg_root=args.seg_root,
            split=split,
            require_local_ct=not args.include_missing_ct,
        )
        candidates_by_split[split] = candidates
        skipped_by_split[split] = skipped

    split_policy = "source train -> train; source val -> test; source test ignored unless local"
    if args.test_from_local_val:
        source_val_as_test = candidates_by_split["val"]
        for entry in source_val_as_test:
            entry["split"] = "test"
        candidates_by_split = {
            "train": candidates_by_split["train"],
            "val": [],
            "test": source_val_as_test + candidates_by_split["test"],
        }
    else:
        split_policy = "source train -> train; source val -> val; source test -> test"

    print("RexGrounding lung nodule CT manifest")
    print(f"  categories : {', '.join(args.categories)}")
    print(f"  manifest   : {args.manifest}")
    for split, candidates in candidates_by_split.items():
        print(f"  {split:5s}      : {len(candidates)} CT scans")
        skipped = skipped_by_split[split]
        if skipped:
            skipped_text = ", ".join(
                f"{reason}={count}" for reason, count in skipped.most_common()
            )
            print(f"             skipped: {skipped_text}")

    if args.dry_run:
        return

    write_manifest(
        manifest_path=args.manifest,
        categories=categories,
        candidates_by_split=candidates_by_split,
        skipped_by_split=skipped_by_split,
        split_policy=split_policy,
    )
    print(f"\nManifest written: {args.manifest}")

    if not args.prepare_nnunet:
        return

    train_val_candidates = candidates_by_split["train"] + candidates_by_split["val"]
    converted, convert_skipped = convert_entries(
        candidates=train_val_candidates,
        images_dir=images_tr,
        labels_dir=labels_tr,
        link_mode=args.link_mode,
        overwrite_labels=args.overwrite_labels,
        max_cases=args.max_cases,
    )

    write_dataset_json(dataset_dir, dataset_full_name, len(converted))

    print("\nPrepared nnU-Net dataset")
    print(f"  dataset      : {dataset_full_name}")
    print(f"  path         : {dataset_dir}")
    print(f"  categories   : {', '.join(args.categories)}")
    print(f"  train cases  : {len(converted)}")
    if args.max_cases:
        print(f"  max cases    : {args.max_cases}")
    if convert_skipped:
        print("  skipped during conversion:")
        for reason, count in convert_skipped.most_common():
            print(f"    {reason}: {count}")

    print("\nSet these before planning/training:")
    print(f"  export nnUNet_raw={args.nnunet_raw}")
    print(f"  export nnUNet_preprocessed={args.nnunet_raw.parent / 'nnUNet_preprocessed'}")
    print(f"  export nnUNet_results={args.nnunet_raw.parent / 'nnUNet_results'}")
    print("\nNext command:")
    print(
        f"  nnUNetv2_plan_and_preprocess -d {args.dataset_id} "
        "--verify_dataset_integrity -c 3d_fullres 2d -np 2 2 --no_pbar"
    )


if __name__ == "__main__":
    main()
