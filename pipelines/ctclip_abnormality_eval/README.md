# CT-CLIP Abnormality Evaluation

This pipeline evaluates a pretrained zero-shot CT-CLIP checkpoint on case-level abnormality detection for:

- `Atelectasis`
- `Lung nodule`
- `Lung opacity`
- `Consolidation`

It supports:

- `ct-rate`: reads ground truth from CT-RATE label CSV files
- `rexgrounding-ct`: derives case-level labels from ReXGroundingCT finding text

## What It Produces

- `predictions.csv`: per-case probabilities and ground-truth labels
- `summary.json`: dataset counts and per-abnormality metrics

## Current Local Data Notes

- Local `data/ct-rate` contains labels/reports CSV files, but no CT volumes right now
- Local ReXGroundingCT subset is available under `data/data_volumes`

## Example

```powershell
.\.venv\Scripts\python.exe pipelines\ctclip_abnormality_eval\pipeline.py `
  --dataset rexgrounding-ct `
  --volume-root data\data_volumes `
  --rex-metadata-json data\Govindu\rexgrounding-ct\dataset.json `
  --checkpoint D:\path\to\CT-CLIP_v2.pt `
  --output-dir outputs\ctclip_rex_eval `
  --device cpu `
  --limit 10
```

## Dry Run

Use dry-run to confirm dataset resolution before running model inference:

```powershell
.\.venv\Scripts\python.exe pipelines\ctclip_abnormality_eval\pipeline.py `
  --dataset rexgrounding-ct `
  --volume-root data\data_volumes `
  --rex-metadata-json data\Govindu\rexgrounding-ct\dataset.json `
  --dry-run
```
