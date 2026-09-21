import csv
from pathlib import Path

import numpy as np

from eval.core.schemas import ClassificationSample


class CTClipClassificationAdapter:
    NPZ_CLASS_ORDER = [
        "atelectasis",
        "lung_nodule",
        "lung_opacity",
        "consolidation",
    ]

    def __init__(self, predictions_csv: str | Path, model_name="ct-clip"):
        self.predictions_path = Path(predictions_csv)
        self.model_name = model_name

    def load(self):
        if self.predictions_path.is_dir():
            return self._load_from_inference_dir(self.predictions_path)

        if self.predictions_path.suffix.lower() == ".csv":
            return self._load_from_csv(self.predictions_path)

        if self.predictions_path.name == "predicted_weights.npz":
            return self._load_from_inference_dir(self.predictions_path.parent)

        raise ValueError(
            "CT-CLIP predictions must be a CSV file, an inference output directory, "
            "or a predicted_weights.npz file."
        )

    def _load_from_csv(self, predictions_csv: Path):
        samples = []

        with predictions_csv.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                y_true = {}
                y_score = {}

                for key, value in row.items():
                    if key.endswith("_gt"):
                        class_key = key[:-3]
                        y_true[class_key] = int(value)

                    if key.endswith("_prob"):
                        class_key = key[:-5]
                        y_score[class_key] = float(value)

                samples.append(
                    ClassificationSample(
                        case_id=row["case_id"],
                        model_name=self.model_name,
                        y_true=y_true,
                        y_score=y_score,
                        dataset=row.get("source", "unknown"),
                        metadata={"volume_path": row.get("volume_path")},
                    )
                )

        return samples

    def _load_from_inference_dir(self, inference_dir: Path):
        labels_path = inference_dir / "labels_weights.npz"
        predictions_path = inference_dir / "predicted_weights.npz"
        accessions_path = inference_dir / "accessions.txt"

        for required_path in (labels_path, predictions_path):
            if not required_path.exists():
                raise FileNotFoundError(
                    f"Expected CT-CLIP inference artifact not found: {required_path}"
                )

        y_true_matrix = self._read_npz_array(labels_path)
        y_score_matrix = self._read_npz_array(predictions_path)

        if y_true_matrix.shape != y_score_matrix.shape:
            raise ValueError(
                "CT-CLIP labels and predictions have different shapes: "
                f"{y_true_matrix.shape} vs {y_score_matrix.shape}"
            )

        if y_true_matrix.ndim != 2:
            raise ValueError(
                f"Expected 2D CT-CLIP arrays, got shape {y_true_matrix.shape}"
            )

        if y_true_matrix.shape[1] != len(self.NPZ_CLASS_ORDER):
            raise ValueError(
                "Unexpected CT-CLIP class dimension. "
                f"Expected {len(self.NPZ_CLASS_ORDER)}, got {y_true_matrix.shape[1]}"
            )

        case_ids = self._load_case_ids(accessions_path, y_true_matrix.shape[0])

        samples = []
        for index, case_id in enumerate(case_ids):
            y_true = {
                class_name: int(round(float(y_true_matrix[index, class_idx])))
                for class_idx, class_name in enumerate(self.NPZ_CLASS_ORDER)
            }
            y_score = {
                class_name: float(y_score_matrix[index, class_idx])
                for class_idx, class_name in enumerate(self.NPZ_CLASS_ORDER)
            }

            samples.append(
                ClassificationSample(
                    case_id=case_id,
                    model_name=self.model_name,
                    y_true=y_true,
                    y_score=y_score,
                    dataset="unknown",
                    metadata={"inference_dir": str(inference_dir)},
                )
            )

        return samples

    @staticmethod
    def _read_npz_array(path: Path):
        loaded = np.load(path)
        if "data" not in loaded:
            raise KeyError(f"Expected 'data' array in {path}")
        return loaded["data"]

    @staticmethod
    def _load_case_ids(accessions_path: Path, expected_count: int):
        raw_ids = []
        if accessions_path.exists():
            raw_ids = [
                line.strip()
                for line in accessions_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        if len(raw_ids) != expected_count:
            return [f"case_{index:05d}" for index in range(expected_count)]

        duplicate_counts = {}
        normalized_ids = []
        for index, raw_id in enumerate(raw_ids):
            base_id = raw_id or f"case_{index:05d}"
            count = duplicate_counts.get(base_id, 0)
            duplicate_counts[base_id] = count + 1
            normalized_ids.append(base_id if count == 0 else f"{base_id}_{count}")

        return normalized_ids
