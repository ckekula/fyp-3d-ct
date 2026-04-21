import time
from huggingface_hub import hf_hub_download
import pandas as pd
import os

repo_id = "ibrahimhamamci/CT-RATE"
directory_name = "dataset/valid_fixed/"
token = os.environ.get("HF_TOKEN")

segmentations_folder_path = r"/home/chest_ct/code/data/segmentations/segmentations"

segmentations = [
    f for f in os.listdir(segmentations_folder_path)
    if os.path.isfile(os.path.join(segmentations_folder_path, f))
]

valid_labels = pd.read_csv("/home/chest_ct/code/data/ct-rate/valid_labels.csv")

files = []
# Iterate over the rows of the dataframe
for index, row in valid_labels.iterrows():
    if row["Lung nodule"] == 1 or row["Lung opacity"] == 1:
        if row["VolumeName"] in segmentations:
            files.append(row["VolumeName"])

failed = []

max_retries = 3
retry_delay = 5

for name in files:
    attempts = 0
    while attempts < max_retries:
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
            break  # If successful, break out of the retry loop

        except Exception as e:
            attempts += 1
            print(f"❌ Attempt {attempts} failed for {name}: {e}")

            if attempts >= max_retries:
                print(f"❌ Failed to download {name} after {max_retries} attempts.")
                failed.append((name, str(e)))
            else:
                print(f"⏳ Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

# Save failures for later retry
pd.DataFrame(failed, columns=["VolumeName", "Error"]).to_csv("failed_valid_downloads.csv", index=False)