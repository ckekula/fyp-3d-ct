import csv
from pathlib import Path
from eval.core.schemas import ClassificationSample


class CTClipClassificationAdapter:
    def __init__(self, predictions_csv: str | Path, model_name="ct-clip"):
        self.predictions_csv = Path(predictions_csv)
        self.model_name = model_name

    def load(self):
        samples = []

        with self.predictions_csv.open("r", encoding="utf-8") as f:
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