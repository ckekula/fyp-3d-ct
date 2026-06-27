from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="rajpurkarlab/ReXGroundingCT",
    repo_type="dataset",
    allow_patterns="segmentations/*.nii.gz",
    local_dir="segmentations",
)