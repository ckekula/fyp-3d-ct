import os
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

json_path = "/home/chest_ct/code/data/rexgrounding-ct/dataset_2_final.json"

existing_ct_dir = Path(
    "/home/chest_ct/code/data/data_volumes/dataset/train_fixed"
)

download_dir = Path(
    "/home/chest_ct/code/data/data_volumes/dataset/train_fixed"
)

download_dir.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------
# Load dataset JSON
# -------------------------------------------------------

with open(json_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)


scan_names = [
    item["name"]
    for item in dataset["train"]
]

print(f"Total scans required: {len(scan_names)}")


# -------------------------------------------------------
# CT-RATE settings
# -------------------------------------------------------

repo_id = "ibrahimhamamci/CT-RATE"


# -------------------------------------------------------
# Helper: find CT-RATE nested path
# -------------------------------------------------------

def get_ct_rate_path(scan_name):
    """
    Example:
    train_1742_c_2.nii.gz

    CT-RATE structure:
    dataset/train/train_1742/train_1742_c/train_1742_c_2.nii.gz
    """

    stem = scan_name.replace(".nii.gz", "")

    # remove last component (_2)
    series_folder = stem.rsplit("_", 1)[0]

    # remove series suffix for patient folder
    patient_folder = series_folder.rsplit("_", 1)[0]

    split = "train" if stem.startswith("train") else "valid"

    return (
        f"dataset/{split}/"
        f"{patient_folder}/"
        f"{series_folder}/"
        f"{scan_name}"
    )


# -------------------------------------------------------
# Download loop
# -------------------------------------------------------

downloaded = []
already_exist = []
failed = []


for scan_name in scan_names:

    local_file = existing_ct_dir / scan_name


    # -----------------------------------------------
    # 1. Check existing flat directory
    # -----------------------------------------------
    if local_file.exists():
        already_exist.append(scan_name)
        continue


    # -----------------------------------------------
    # 2. Download from CT-RATE
    # -----------------------------------------------
    hf_path = get_ct_rate_path(scan_name)

    print(f"Downloading: {scan_name}")
    print(f"HF path: {hf_path}")

    try:

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=hf_path,
            local_dir="ct_rate_cache",
            resume_download=True
        )


        # Move/copy into flat directory
        shutil.copy(
            downloaded_path,
            local_file
        )

        downloaded.append(scan_name)


    except Exception as e:

        print(f"FAILED: {scan_name}")
        print(e)

        failed.append(scan_name)



# -------------------------------------------------------
# Summary
# -------------------------------------------------------

print("\n========== SUMMARY ==========")

print(f"Already existed : {len(already_exist)}")
print(f"Downloaded      : {len(downloaded)}")
print(f"Failed          : {len(failed)}")


if failed:
    print("\nFailed scans:")
    for x in failed:
        print(x)