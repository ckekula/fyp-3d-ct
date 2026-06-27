from pathlib import Path
import json
import shutil

# -----------------------------
# Paths
# -----------------------------
json_path = Path("/home/chest_ct/code/data/rexgrounding-ct/dataset_2d.json")

dataset_root = Path("/home/chest_ct/code/data/data_volumes/dataset")

search_dirs = [
    Path("/home/chest_ct/code/data/data_volumes/dataset/valid_fixed"),
    Path("/home/chest_ct/code/data/data_volumes/dataset/train_fixed"),
]

output_root = Path("/home/chest_ct/code/models/nnu-net/storage/nnUNet_raw/Dataset104_Nodules")

images_tr = output_root / "imagesTr"
images_ts = output_root / "imagesTs"

images_tr.mkdir(parents=True, exist_ok=True)
images_ts.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Load JSON
# -----------------------------
with open(json_path, "r") as f:
    data = json.load(f)

train_files = data["train"]
test_files = data["test"]

# -----------------------------
# Build lookup of all nifti files
# -----------------------------
all_nifti_files = {}

for folder in search_dirs:
    for file_path in folder.rglob("*.nii.gz"):
        all_nifti_files[file_path.name] = file_path

print(f"Found {len(all_nifti_files)} total NIfTI files")

# -----------------------------
# Move function
# -----------------------------
def move_files(file_list, destination_folder, split_name):
    moved = 0
    missing = []

    for filename in file_list:
        src = all_nifti_files.get(filename)

        if src is None:
            missing.append(filename)
            continue

        dst = destination_folder / filename

        if dst.exists():
            print(f"[SKIP] Already exists: {dst}")
            continue

        shutil.copy2(str(src), str(dst))
        moved += 1
        print(f"[MOVED] {src} -> {dst}")

    print(f"\n{split_name}: moved {moved}/{len(file_list)} files")

    if missing:
        print(f"{split_name}: missing {len(missing)} files")
        for m in missing[:20]:
            print("  Missing:", m)

        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")


# -----------------------------
# Move train and test files
# -----------------------------
move_files(train_files, images_tr, "train")
move_files(test_files, images_ts, "test")