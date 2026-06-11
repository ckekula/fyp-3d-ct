from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="rajpurkarlab/ReXGroundingCT",
    repo_type="dataset",
    subfolder="segmentations",
    filename="train_67_a_2.nii.gz",
    local_dir="data_volumes",
)