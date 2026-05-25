#!/usr/bin/env python
"""Quick validation without heavy dependencies."""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]

print("Quick Pipeline Validation")
print("=" * 60)

# Check metadata
metadata_path = ROOT / "data" / "rexgrounding-ct" / "dataset.json"
print(f"1. Metadata file: {metadata_path}")
print(f"   Exists: {metadata_path.exists()}")

if metadata_path.exists():
    with metadata_path.open() as f:
        data = json.load(f)
    print(f"   Train cases: {len(data.get('train', []))}")
    print(f"   Valid cases: {len(data.get('valid', []))}")
    
    if data.get('train'):
        sample = data['train'][0]
        print(f"\n   Sample case: {sample['name']}")
        print(f"   Findings: {list(sample.get('findings', {}).values())[:2]}...")

# Check pipeline files
pipeline_files = [
    "pipeline.py",
    "dataset_adapter.py",
    "prompts.py",
    "visualize.py",
    "test_pipeline.py",
]

print(f"\n2. Pipeline files:")
for fname in pipeline_files:
    fpath = Path(__file__).parent / fname
    print(f"   {fname}: {'✓' if fpath.exists() else '✗'}")

# Import and test prompts
print(f"\n3. Target diseases defined:")
sys.path.insert(0, str(Path(__file__).parent))

try:
    from prompts import TARGET_DISEASES
    for disease, prompts in TARGET_DISEASES.items():
        print(f"   - {disease} ({len(prompts)} prompts)")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("✓ Basic validation complete")
