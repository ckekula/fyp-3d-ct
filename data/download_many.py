from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
import json

with open("/home/chest_ct/code/data/rexgrounding-ct/dataset.json", "r") as file:
    rex_data = json.load(file)

scans_with_only_2d = []

for split in ["train", "val", "test"]:
    df = pd.DataFrame(rex_data[split])

    for _, row in df.iterrows():
        cats = set(row["categories"].values())

        if cats == {"2d"}:
            scans_with_only_2d.append(row)

scans_with_only_2d_df = pd.DataFrame(scans_with_only_2d)
scan_names = scans_with_only_2d_df["name"].tolist()

local_dir = Path("data_volumes")

downloaded_files = []
failed_files = []

for i, scan_name in enumerate(scan_names, start=1):
    try:
        # Example:
        # scan_name = train_1168_a_2.nii.gz
        # patient_id = train_1168
        # study_id   = train_1168_a

        name_no_ext = scan_name.replace(".nii.gz", "")

        parts = name_no_ext.split("_")
        patient_id = f"{parts[0]}_{parts[1]}"        # train_1168
        study_id = f"{parts[0]}_{parts[1]}_{parts[2]}"  # train_1168_a

        subfolder = f"dataset/valid_fixed/{patient_id}/{study_id}"

        local_path = hf_hub_download(
            repo_id="rajpurkarlab/ReXGroundingCT",
            repo_type="dataset",
            subfolder="segmentations",
            filename=scan_name,
            local_dir=local_dir,
        )

        downloaded_files.append(local_path)
        print(f"[{i}/{len(scan_names)}] Downloaded: {scan_name}")
        print(f"    Saved to: {local_path}")

    except Exception as e:
        failed_files.append({
            "name": scan_name,
            "error": str(e),
        })

        print(f"[{i}/{len(scan_names)}] FAILED: {scan_name}")
        print(f"    Error: {e}")
        continue

print("\nDownload summary")
print(f"Requested:   {len(scan_names)}")
print(f"Downloaded: {len(downloaded_files)}")
print(f"Failed:     {len(failed_files)}")

if failed_files:
    failed_df = pd.DataFrame(failed_files)
    failed_df.to_csv("failed_downloads.csv", index=False)
    print("Saved failed list to failed_downloads.csv")