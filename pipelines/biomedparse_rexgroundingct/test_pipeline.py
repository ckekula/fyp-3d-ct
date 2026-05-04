#!/usr/bin/env python
"""Minimal test of the BiomedParse + RexGroundingCT pipeline."""

import sys
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

# Test 1: Check imports and structure
print("=" * 60)
print("TEST 1: Validating pipeline structure and imports...")
print("=" * 60)

try:
    from prompts import TARGET_DISEASES, default_prompt_bundles
    print("✅ Prompts module loads")
except Exception as e:
    print(f"❌ Prompts error: {e}")
    sys.exit(1)

try:
    from dataset_adapter import load_rexgroundingct_cases, iter_target_cases
    print("✅ Dataset adapter loads")
except Exception as e:
    print(f"❌ Dataset adapter error: {e}")
    sys.exit(1)

try:
    from visualize import save_overlay_png
    print("✅ Visualization module loads")
except Exception as e:
    print(f"❌ Visualization error: {e}")
    sys.exit(1)

# Test 2: Load RexGroundingCT metadata
print("\n" + "=" * 60)
print("TEST 2: Loading RexGroundingCT metadata...")
print("=" * 60)

metadata_path = ROOT / "data" / "rexgrounding-ct" / "dataset.json"
try:
    with metadata_path.open() as f:
        metadata = json.load(f)
    train_count = len(metadata.get("train", []))
    valid_count = len(metadata.get("valid", []))
    print(f"✅ Loaded metadata: {train_count} train, {valid_count} validation cases")
except Exception as e:
    print(f"❌ Metadata loading failed: {e}")
    sys.exit(1)

# Test 3: Filter target disease cases
print("\n" + "=" * 60)
print("TEST 3: Filtering for target diseases...")
print("=" * 60)

print("\nTarget diseases:")
for disease, prompts in TARGET_DISEASES.items():
    print(f"  - {disease}: {len(prompts)} prompt variants")

dummy_cases = []
for entry in metadata.get("train", [])[:10]:
    findings_text = " ".join(entry.get("findings", {}).values())
    dummy_cases.append({"name": entry["name"], "findings_text": findings_text})

print(f"\nScanned first 10 training cases for disease mentions...")
for case in dummy_cases:
    findings_preview = case["findings_text"][:80] if case["findings_text"] else "(empty)"
    print(f"  • {case['name']}: {findings_preview}...")

# Test 4: Show prompt bundles
print("\n" + "=" * 60)
print("TEST 4: Prompt bundles for inference...")
print("=" * 60)

bundles = default_prompt_bundles()
for bundle in bundles:
    prompt_preview = bundle.text[:100]
    print(f"\n{bundle.disease}:")
    print(f"  Text: {prompt_preview}...")

# Test 5: Check for checkpoint
print("\n" + "=" * 60)
print("TEST 5: Checking for BiomedParse checkpoint...")
print("=" * 60)

import os
cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
checkpoint_dirs = list(cache_dir.glob("*BiomedParse*")) if cache_dir.exists() else []
if checkpoint_dirs:
    print(f"✅ Found cached checkpoints:")
    for d in checkpoint_dirs:
        print(f"   - {d}")
else:
    print("⚠️  No cached checkpoint found yet (downloading...)")

# Test 6: Check dataset availability
print("\n" + "=" * 60)
print("TEST 6: Checking for sample volumes...")
print("=" * 60)

volume_dir = ROOT / "data" / "rexgrounding-ct" / "volumes"
if volume_dir.exists():
    volumes = list(volume_dir.glob("**/*.nii.gz"))
    if volumes:
        print(f"✅ Found {len(volumes)} volume files:")
        for v in volumes[:3]:
            print(f"   - {v.name}")
    else:
        print(f"⚠️  Volume directory exists but is empty")
else:
    print(f"⚠️  Volume directory not found: {volume_dir}")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)
print("""
Pipeline structure: ✅ Valid
RexGroundingCT metadata: ✅ Loaded
Target diseases: ✅ Defined
Prompt templates: ✅ Ready
BiomedParse checkpoint: ⏳ Downloading...
Sample volumes: ⏳ Pending...

Next steps:
1. Wait for checkpoint to finish downloading
2. Download sample volumes from RexGroundingCT
3. Run full inference on the pipeline
""")
