from huggingface_hub import hf_hub_download
from pathlib import Path
import pandas as pd
import json
import shutil

# -------------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------------
with open("/home/chest_ct/code/data/rexgrounding-ct/dataset_2_last.json", "r") as file:
    rex_data = json.load(file)

scan_names = []

for split in ["train", "test"]:
    df = pd.DataFrame(rex_data[split])
    scan_names.extend(df["name"].tolist())


# -------------------------------------------------------------------------
# Download settings
# -------------------------------------------------------------------------
local_dir = Path("data_volumes")
target_dir = local_dir / "dataset" / "train_fixed"
target_dir.mkdir(parents=True, exist_ok=True)

downloaded_files = []
skipped_files = []
failed_files = []

# -------------------------------------------------------------------------
# Download loop
# -------------------------------------------------------------------------
for i, scan_name in enumerate(scan_names, start=1):
    final_path = target_dir / scan_name

    # Skip if already downloaded
    if final_path.exists():
        skipped_files.append(str(final_path))
        print(f"[{i}/{len(scan_names)}] Skipped: {scan_name} (already exists)")
        continue

    try:
        # Example:
        # scan_name = train_1168_a_2.nii.gz
        # patient_id = train_1168
        # study_id   = train_1168_a

        name_no_ext = scan_name.replace(".nii.gz", "")

        parts = name_no_ext.split("_")
        patient_id = f"{parts[0]}_{parts[1]}"
        study_id = f"{parts[0]}_{parts[1]}_{parts[2]}"

        subfolder = f"dataset/valid_fixed/{patient_id}/{study_id}"

        local_path = Path(
            hf_hub_download(
                repo_id="ibrahimhamamci/CT-RATE",
                repo_type="dataset",
                subfolder=subfolder,
                filename=scan_name,
                local_dir=local_dir,
            )
        )

        # Move to flattened directory
        shutil.move(local_path, final_path)

        downloaded_files.append(str(final_path))

        print(f"[{i}/{len(scan_names)}] Downloaded: {scan_name}")
        print(f"    Saved to: {final_path}")

    except Exception as e:
        failed_files.append({
            "name": scan_name,
            "error": str(e),
        })

        print(f"[{i}/{len(scan_names)}] FAILED: {scan_name}")
        print(f"    Error: {e}")

print("\nDownload summary")
print(f"Requested:   {len(scan_names)}")
print(f"Downloaded:  {len(downloaded_files)}")
print(f"Skipped:     {len(skipped_files)}")
print(f"Failed:      {len(failed_files)}")

if failed_files:
    failed_df = pd.DataFrame(failed_files)
    failed_df.to_csv("failed_downloads.csv", index=False)
    print("Saved failed list to failed_downloads.csv")
