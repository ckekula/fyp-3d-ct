from huggingface_hub import hf_hub_download

# CT-RATE file location
repo_id = "ibrahimhamamci/CT-RATE"

file_path = (
    "dataset/valid/"
    "valid_342/"
    "valid_342_a/"
    "valid_342_a_2.nii.gz"
)

# Download
local_path = hf_hub_download(
    repo_id=repo_id,
    repo_type="dataset",
    filename=file_path,
    local_dir="data_volumes",
    resume_download=True
)

print("Downloaded file:")
print(local_path)