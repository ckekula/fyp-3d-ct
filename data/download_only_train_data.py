from huggingface_hub import hf_hub_download
import pandas as pd
import os

repo_id = "ibrahimhamamci/CT-RATE"
directory_name = "dataset/train_fixed/"
token = os.environ.get("HF_TOKEN")

data = pd.read_csv("train_labels.csv")

segmentations_folder_path = r"C:\Users\chamu\D\UOR\S7\FYP\fyp\data\ct-rate\data_segmentations\segmentations"

files = [
    f for f in os.listdir(segmentations_folder_path)
    if os.path.isfile(os.path.join(segmentations_folder_path, f))
]

failed = []

for name in data["VolumeName"]: # Use files to download specific files
    try:
        if not isinstance(name, str):
            raise ValueError("VolumeName is not a string")

        parts = name.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unexpected filename format: {name}")

        folder = f"{parts[0]}_{parts[1]}"
        subfolder = f"{directory_name}{folder}/{folder}_{parts[2]}"

        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            subfolder=subfolder,
            filename=name,
            local_dir="data_volumes",
            token=token,
        )

        print(f"✅ Downloaded {name}")

    except Exception as e:
        print(f"❌ Failed {name}: {e}")
        failed.append((name, str(e)))

# Save failures for later retry
pd.DataFrame(failed, columns=["VolumeName", "Error"]).to_csv("failed_downloads.csv", index=False)
