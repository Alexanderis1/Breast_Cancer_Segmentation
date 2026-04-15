"""
Ground Truth Verification Script — Duke Breast Cancer MRI Dataset
=================================================================
Run this BEFORE training to validate that:
  1. DICOM files are sorted correctly (InstanceNumber vs. alphabetical)
  2. Bounding box z-coordinates are within the InstanceNumber range of exported DICOMs
  3. Each patient's tumor is actually covered by the exported slices

Usage:
    python verify_ground_truth.py
    python verify_ground_truth.py --export-dir exported_patients
    python verify_ground_truth.py --export-dir exported_patients --split train --max-patients 50
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import pydicom
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# DICOM utilities
# ─────────────────────────────────────────────────────────────────────────────

def get_instance_number(dcm_path: Path) -> int:
    """Read InstanceNumber tag from DICOM header (no pixel data loaded)."""
    try:
        dcm = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        return int(getattr(dcm, "InstanceNumber", -1))
    except Exception:
        return -1


def load_dicom_instances(mri_dir: Path) -> list[tuple[int, Path]]:
    """
    Return sorted list of (instance_number, path) for all DICOMs in mri_dir.
    Files with unreadable InstanceNumber get -1.
    """
    dcm_files = list(mri_dir.glob("**/*.dcm"))
    tagged = [(get_instance_number(p), p) for p in dcm_files]
    tagged.sort(key=lambda x: x[0])
    return tagged


def alpha_order(paths: list[Path]) -> list[Path]:
    """Sort paths the same way the current dataset code does (alphabetical)."""
    return sorted(paths)


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_patient(patient_dir: Path) -> dict:
    result = {
        "patient_id": patient_dir.name,
        "status": "ok",
        "num_dicoms": 0,
        "has_negative_instances": False,
        "alpha_vs_instance_mismatch": False,
        "instance_min": None,
        "instance_max": None,
        "bbox_z_min": None,
        "bbox_z_max": None,
        "z_covered": None,   # True/False/None
        "overlap_pct": None,
        "warnings": [],
    }

    # ── load metadata ────────────────────────────────────────────────────────
    data_file = patient_dir / "patient_data.json"
    if not data_file.exists():
        result["status"] = "missing_json"
        result["warnings"].append("patient_data.json not found")
        return result

    try:
        with open(data_file) as f:
            metadata = json.load(f)
    except Exception as e:
        result["status"] = "json_error"
        result["warnings"].append(f"JSON parse error: {e}")
        return result

    annotations = metadata.get("annotations", [])
    if not annotations:
        result["status"] = "no_annotations"
        result["warnings"].append("No annotations in JSON")
        return result

    ann = annotations[0]
    bbox_3d = ann.get("bounding_box_3d", {})
    z_min_gt = bbox_3d.get("start_slice")
    z_max_gt = bbox_3d.get("end_slice")

    if z_min_gt is None or z_max_gt is None:
        result["status"] = "no_z_coords"
        result["warnings"].append("start_slice / end_slice missing from annotation")
        return result

    result["bbox_z_min"] = float(z_min_gt)
    result["bbox_z_max"] = float(z_max_gt)

    # ── load DICOMs ──────────────────────────────────────────────────────────
    mri_dir = patient_dir / "MRI_DICOM_sample"
    if not mri_dir.exists():
        result["status"] = "missing_mri_dir"
        result["warnings"].append("MRI_DICOM_sample/ directory not found")
        return result

    tagged = load_dicom_instances(mri_dir)
    if not tagged:
        result["status"] = "no_dicoms"
        result["warnings"].append("No .dcm files found in MRI_DICOM_sample/")
        return result

    result["num_dicoms"] = len(tagged)

    instances = [t for t, _ in tagged]
    paths = [p for _, p in tagged]

    # ── check for unreadable InstanceNumbers ─────────────────────────────────
    neg_count = sum(1 for i in instances if i < 0)
    if neg_count > 0:
        result["has_negative_instances"] = True
        result["warnings"].append(
            f"{neg_count}/{len(instances)} DICOMs have unreadable InstanceNumber"
        )

    valid_instances = [i for i in instances if i >= 0]
    if valid_instances:
        result["instance_min"] = min(valid_instances)
        result["instance_max"] = max(valid_instances)
    else:
        result["warnings"].append("All InstanceNumbers unreadable; cannot validate z-coverage")
        return result

    # ── check alphabetical vs. InstanceNumber sort mismatch ──────────────────
    alpha_paths = alpha_order(paths)
    instance_paths = paths  # already sorted by instance
    if alpha_paths != instance_paths:
        result["alpha_vs_instance_mismatch"] = True
        result["warnings"].append(
            "Alphabetical order != InstanceNumber order — slices will be scrambled in current code"
        )

    # ── z-coverage check ─────────────────────────────────────────────────────
    inst_min = result["instance_min"]
    inst_max = result["instance_max"]

    # Overlap between [inst_min, inst_max] and [z_min_gt, z_max_gt]
    overlap_start = max(inst_min, z_min_gt)
    overlap_end = min(inst_max, z_max_gt)
    gt_span = z_max_gt - z_min_gt + 1
    overlap = max(0, overlap_end - overlap_start + 1)

    result["overlap_pct"] = round(100.0 * overlap / gt_span, 1) if gt_span > 0 else 0.0
    result["z_covered"] = (overlap_pct := result["overlap_pct"]) >= 50.0

    if overlap_pct < 100.0:
        severity = "ERROR" if overlap_pct < 50.0 else "WARNING"
        result["warnings"].append(
            f"{severity}: bbox z=[{z_min_gt:.0f},{z_max_gt:.0f}] only "
            f"{overlap_pct:.1f}% covered by exported instances [{inst_min},{inst_max}]"
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

HEADER = (
    f"{'Patient':<20} {'#DICOMs':>7} {'Inst range':>12} "
    f"{'BBox z':>12} {'Overlap%':>9} {'Sort OK':>8} {'Warnings'}"
)
SEP = "-" * 100


def format_row(r: dict) -> str:
    pid = r["patient_id"][:19]
    n = r["num_dicoms"]
    inst = (
        f"[{r['instance_min']},{r['instance_max']}]"
        if r["instance_min"] is not None else "N/A"
    )
    bbox_z = (
        f"[{r['bbox_z_min']:.0f},{r['bbox_z_max']:.0f}]"
        if r["bbox_z_min"] is not None else "N/A"
    )
    overlap = f"{r['overlap_pct']:.1f}%" if r["overlap_pct"] is not None else "N/A"
    sort_ok = "YES" if not r["alpha_vs_instance_mismatch"] else "NO"
    warns = "; ".join(r["warnings"]) if r["warnings"] else ""
    return (
        f"{pid:<20} {n:>7} {inst:>12} {bbox_z:>12} {overlap:>9} {sort_ok:>8}  {warns}"
    )


def print_summary(results: list[dict], split: str, out_lines: list[str]) -> None:
    def p(line=""):
        print(line)
        out_lines.append(line)

    p(f"\n{'='*100}")
    p(f"SPLIT: {split.upper()}   ({len(results)} patients)")
    p(f"{'='*100}")
    p(HEADER)
    p(SEP)

    statuses = defaultdict(int)
    sort_mismatches = 0
    coverage_full = 0
    coverage_partial = 0
    coverage_none = 0

    for r in results:
        p(format_row(r))
        statuses[r["status"]] += 1
        if r["alpha_vs_instance_mismatch"]:
            sort_mismatches += 1
        if r["overlap_pct"] is not None:
            if r["overlap_pct"] >= 100.0:
                coverage_full += 1
            elif r["overlap_pct"] >= 50.0:
                coverage_partial += 1
            else:
                coverage_none += 1

    p()
    p("SUMMARY")
    p(SEP)
    p(f"  Total patients:         {len(results)}")
    p(f"  Sort mismatch (Bug 2):  {sort_mismatches} patients — alphabetical != InstanceNumber order")
    p(f"  Z-coverage FULL (100%): {coverage_full}")
    p(f"  Z-coverage PARTIAL (50-99%): {coverage_partial}")
    p(f"  Z-coverage NONE (<50%): {coverage_none}  ← these patients have z-Bug 1")
    p()
    for st, count in sorted(statuses.items()):
        p(f"  status={st}: {count}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Verify Duke dataset ground truth quality")
    parser.add_argument("--export-dir", default="exported_patients",
                        help="Path to exported_patients directory")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both",
                        help="Which split(s) to verify")
    parser.add_argument("--max-patients", type=int, default=None,
                        help="Cap number of patients per split (for quick checks)")
    parser.add_argument("--output", default="ground_truth_report.txt",
                        help="Path to save the text report")
    return parser.parse_args()


def main():
    args = parse_args()
    export_dir = Path(args.export_dir)

    if not export_dir.exists():
        print(f"ERROR: export directory not found: {export_dir}")
        sys.exit(1)

    splits = []
    if args.split in ("train", "both"):
        splits.append("train")
    if args.split in ("test", "both"):
        splits.append("test")

    out_lines: list[str] = []
    all_results: dict[str, list[dict]] = {}

    for split in splits:
        split_dir = export_dir / split
        if not split_dir.exists():
            print(f"WARNING: split directory not found: {split_dir}")
            continue

        patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        if args.max_patients:
            patient_dirs = patient_dirs[: args.max_patients]

        print(f"\nAnalyzing {split}: {len(patient_dirs)} patients ...", flush=True)
        results = []
        for i, pd in enumerate(patient_dirs):
            r = analyze_patient(pd)
            results.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(patient_dirs)} done", flush=True)

        all_results[split] = results
        print_summary(results, split, out_lines)

    # ── save report ──────────────────────────────────────────────────────────
    report_path = Path(args.output)
    report_path.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\nReport saved to: {report_path}")

    # ── overall verdict ───────────────────────────────────────────────────────
    total_mismatch = sum(
        sum(1 for r in res if r["alpha_vs_instance_mismatch"])
        for res in all_results.values()
    )
    total_no_coverage = sum(
        sum(1 for r in res if r["overlap_pct"] is not None and r["overlap_pct"] < 50.0)
        for res in all_results.values()
    )

    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    if total_mismatch > 0:
        print(f"  BUG 2 (sort order): {total_mismatch} patients affected")
        print("    → Fix: sort DICOMs by InstanceNumber in Duke3DBBoxDatasetV2")
    else:
        print("  BUG 2 (sort order): NOT detected (alphabetical == instance order)")

    if total_no_coverage > 0:
        print(f"  BUG 1 (z-coverage): {total_no_coverage} patients have <50% bbox coverage")
        print("    → Fix: use InstanceNumber range for z-remapping in Duke3DBBoxDatasetV2")
    else:
        print("  BUG 1 (z-coverage): NOT detected (all patients have >=50% bbox coverage)")
    print("=" * 60)


if __name__ == "__main__":
    main()
