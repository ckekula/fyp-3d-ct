"""
repair_checkpoint.py
Repairs a corrupted `_patch_checkpoint_{tag}.json` produced by patch_extraction.py
after memory corruption (suspected hardware — see conversation) wrote garbled
bytes into it. Corruption has been observed both inside string fields (scan_id)
and inside plain numeric fields (coords), and can break json's own delimiter
parsing, so json.loads (even with strict=False) is not used at all here.

Strategy:
  1. Locate each of the 4 top-level arrays (done_scan_ids, labels, scan_ids,
     coords) by their fixed key order in the file (as always written by
     _Checkpoint._save()), and slice out each array's raw text region directly
     — this sidesteps needing the whole file to be valid JSON.
  2. Parse each region leniently: split/match each element independently, and
     mark elements that don't parse as an expected token (int, quoted string,
     3-int coordinate triple) as corrupted, WITHOUT letting one bad element
     abort parsing of the rest.
  3. Drop corrupted records at their original indices (labels/scan_ids/coords
     stay positionally aligned with the scratch .bin file), and drop any
     corrupted done_scan_ids entries so those scans get safely re-extracted.
  4. Rewrite the scratch .bin file, physically dropping the same record
     indices, so positional alignment with the repaired checkpoint arrays
     (which build_patch_matrix relies on) is preserved.

This can only catch corruption that breaks a token's *shape* (e.g. a control
byte or letter where a digit belongs). A bit-flip that still looks like a
valid token (e.g. one digit changed to another digit) is invisible to this
check — see the printed caveat before trusting the "clean" report.

Defaults to a dry run (report only). Pass --apply to actually write changes.
Run from the project root with the venv active, e.g.:
    python repair_checkpoint.py --tag abnormal            # dry run
    python repair_checkpoint.py --tag abnormal --apply    # apply fix
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

# Adjust this import if your project layout differs.
from lc_ksvd.config import N_FEATURES, PATCHES_DIR

RECORD_BYTES = N_FEATURES * 4  # float32

VALID_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
COORD_TRIPLE_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")
BRACKET_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")


def _slice_between(raw: str, start_key: str, end_key: str) -> str:
    """Return the raw text of the array value for `start_key`, up to (not
    including) the next key `end_key`. Relies on the fixed key order that
    _Checkpoint._save() always writes: done_scan_ids, labels, scan_ids, coords."""
    start = raw.index(start_key) + len(start_key)
    start = raw.index("[", start) + 1
    end = raw.index(end_key, start)
    end = raw.rindex("]", start, end)
    return raw[start:end]


def _parse_strings(region: str) -> list[str]:
    """Extract quoted string contents in order, tolerating raw bytes inside."""
    return [m.group(1) for m in STRING_RE.finditer(region)]


def _parse_ints(region: str) -> list[tuple[int | None, str]]:
    """Split a flat comma-separated int list; return (value_or_None, raw_token)."""
    out = []
    for tok in region.split(","):
        tok = tok.strip()
        if re.fullmatch(r"-?\d+", tok):
            out.append((int(tok), tok))
        else:
            out.append((None, tok))
    return out


def _parse_coord_triples(region: str) -> list[tuple[tuple[int, int, int] | None, str]]:
    """Extract each [a, b, c] bracket group in order; return (triple_or_None, raw)."""
    out = []
    for m in BRACKET_GROUP_RE.finditer(region):
        inner = m.group(1)
        tm = COORD_TRIPLE_RE.match(inner)
        if tm:
            out.append(((int(tm.group(1)), int(tm.group(2)), int(tm.group(3))), inner))
        else:
            out.append((None, inner))
    return out


def is_valid_id(s: str) -> bool:
    return isinstance(s, str) and bool(VALID_ID_RE.match(s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True, help="scratch_tag, e.g. 'abnormal' or 'normal'")
    parser.add_argument("--apply", action="store_true", help="actually write repaired files (default: dry run)")
    args = parser.parse_args()

    checkpoint_path = PATCHES_DIR / f"_patch_checkpoint_{args.tag}.json"
    scratch_path = PATCHES_DIR / f"_patch_scratch_{args.tag}.bin"

    if not checkpoint_path.exists():
        sys.exit(f"No checkpoint found at {checkpoint_path}")

    raw = checkpoint_path.read_text(errors="surrogateescape")

    done_region = _slice_between(raw, '"done_scan_ids"', '"labels"')
    labels_region = _slice_between(raw, '"labels"', '"scan_ids"')
    scanids_region = _slice_between(raw, '"scan_ids"', '"coords"')
    # coords is the last key — slice to the final closing bracket of the file.
    coords_start = raw.index('"coords"')
    coords_start = raw.index("[", coords_start) + 1
    coords_region = raw[coords_start:raw.rindex("]")]

    done_scan_ids = _parse_strings(done_region)
    labels_parsed = _parse_ints(labels_region)
    scan_ids_parsed = _parse_strings(scanids_region)
    coords_parsed = _parse_coord_triples(coords_region)

    n = len(labels_parsed)
    if not (len(scan_ids_parsed) == n and len(coords_parsed) == n):
        sys.exit(
            f"Array length mismatch after tolerant parse: labels={n} "
            f"scan_ids={len(scan_ids_parsed)} coords={len(coords_parsed)}. "
            f"The corruption broke element boundaries themselves (not just "
            f"content) — this needs manual inspection, auto-repair isn't safe."
        )

    # --- Find corrupted entries, independently per field, by position ---
    bad_indices: set[int] = set()
    for i, (val, raw_tok) in enumerate(labels_parsed):
        if val is None:
            bad_indices.add(i)
    for i, sid in enumerate(scan_ids_parsed):
        if not is_valid_id(sid):
            bad_indices.add(i)
    for i, (val, raw_tok) in enumerate(coords_parsed):
        if val is None:
            bad_indices.add(i)

    bad_done_ids = [sid for sid in done_scan_ids if not is_valid_id(sid)]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"  scans done:        {len(done_scan_ids)}")
    print(f"  patches recorded:  {n}")
    print(f"  corrupted patch records (any field): {len(bad_indices)}")
    if bad_indices:
        for i in sorted(bad_indices):
            print(f"    idx {i}: label={labels_parsed[i]!r} "
                  f"scan_id={scan_ids_parsed[i]!r} coord={coords_parsed[i]!r}")
    print(f"  corrupted done_scan_ids entries: {len(bad_done_ids)}")
    if bad_done_ids:
        print(f"    {[repr(s) for s in bad_done_ids]}")
    print(
        "\nNote: only *detectable* corruption (values that fail to parse as "
        "valid tokens) can be caught this way. A bit-flip that still looks "
        "like a plausible number/string is invisible to this check. This "
        "repair removes known-bad records; it cannot prove the rest of the "
        "file is uncorrupted."
    )

    if not bad_indices and not bad_done_ids:
        print("\nNo corrupted entries found by these checks — the structural "
              "parse error may be coming from something this script doesn't "
              "cover. Stop and inspect manually rather than applying.")
        return

    if not args.apply:
        print("\nDry run only — rerun with --apply to write the repaired "
              "checkpoint and scratch file.")
        return

    # --- Apply repair ---
    keep_indices = [i for i in range(n) if i not in bad_indices]

    new_labels = [labels_parsed[i][0] for i in keep_indices]
    new_scan_ids = [scan_ids_parsed[i] for i in keep_indices]
    new_coords = [list(coords_parsed[i][0]) for i in keep_indices]
    new_done_scan_ids = sorted(set(done_scan_ids) - set(bad_done_ids))

    if bad_indices:
        print(
            "\nWARNING: dropped patch records are removed individually — if "
            "the scan_id field itself was the corrupted part of a record, "
            "that scan may still be marked 'done' (its other, uncorrupted "
            "patches remain) even though this one patch was silently "
            "dropped. This is treated as an isolated single-record loss, "
            "not grounds to re-run the whole scan, since the corruption "
            "pattern seen so far looks like scattered single-point errors "
            "rather than whole-scan failures. If you'd rather be "
            "conservative and re-run any scan that had ANY corrupted "
            "record, tell me and I'll adjust this."
        )

    # Rewrite scratch .bin, keeping only surviving record indices, to stay
    # positionally aligned with the repaired checkpoint arrays.
    if scratch_path.exists():
        backup_bin = scratch_path.with_suffix(".bin.bak")
        shutil.copy2(scratch_path, backup_bin)
        with open(scratch_path, "rb") as f:
            raw_bin = f.read()
        expected_len = n * RECORD_BYTES
        if len(raw_bin) < expected_len:
            sys.exit(
                f"Scratch file shorter than checkpoint implies "
                f"({len(raw_bin)} bytes < {expected_len} expected) — "
                f"do not proceed automatically, inspect manually."
            )
        new_bin = bytearray()
        for i in keep_indices:
            start = i * RECORD_BYTES
            new_bin += raw_bin[start:start + RECORD_BYTES]
        tmp_bin = scratch_path.with_suffix(".bin.tmp")
        tmp_bin.write_bytes(new_bin)
        tmp_bin.replace(scratch_path)
        print(f"  scratch file repaired ({len(raw_bin)} -> {len(new_bin)} bytes); "
              f"backup at {backup_bin}")
    else:
        print("  no scratch .bin file found (nothing to realign)")

    backup_json = checkpoint_path.with_suffix(".json.bak")
    shutil.copy2(checkpoint_path, backup_json)

    repaired = {
        "done_scan_ids": new_done_scan_ids,
        "labels": new_labels,
        "scan_ids": new_scan_ids,
        "coords": new_coords,
    }
    tmp_json = checkpoint_path.with_suffix(".json.tmp")
    tmp_json.write_text(json.dumps(repaired))
    tmp_json.replace(checkpoint_path)

    print(f"\nRepaired. Backup of original checkpoint at {backup_json}.")
    print(f"Dropped {len(bad_indices)} patch record(s) and "
          f"{len(bad_done_ids)} corrupted done_scan_ids entry(ies).")
    print(f"Remaining: {len(new_done_scan_ids)} scans / {len(new_labels)} patches.")


if __name__ == "__main__":
    main()