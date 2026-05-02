from huggingface_hub import hf_hub_download, list_repo_files
import os
import pandas as pd

repo_id = "rajpurkarlab/ReXGroundingCT"
token = os.environ.get("HF_TOKEN")

all_files = list_repo_files(
    repo_id=repo_id,
    repo_type="dataset",
    token=token,
)

seg_files = [
    f for f in all_files
    if f.startswith("segmentations/") and f.endswith(".nii.gz")
]

print(f"Found {len(seg_files)} segmentation files")

failed = []

# 3. Download using FULL PATH as filename
for file_path in seg_files[1000:]:
    try:
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=file_path,
            local_dir="segmentations",
            resume_download=True,
            token=token,
        )

        print(f"✅ Downloaded {file_path}")

    except Exception as e:
        print(f"❌ Failed {file_path}: {e}")
        failed.append((file_path, str(e)))

# 4. Save failures
if failed:
    pd.DataFrame(
        failed, columns=["file", "error"]
    ).to_csv("failed_segmentations.csv", index=False)
