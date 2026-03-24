import argparse
import json
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize MRI slices with ground-truth 3D bbox for 5 patients."
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        default="exported_patients",
        help="Path to exported patients directory.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=5,
        help="Number of patients to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="gt_bbox_visualizations",
        help="Directory to save generated figures.",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dicom_files(mri_dir):
    files = sorted(mri_dir.glob("**/*.dcm"))
    if not files:
        return []

    # Prefer sorting by InstanceNumber when available for stable slice order.
    def sort_key(fp):
        try:
            dcm = pydicom.dcmread(str(fp), stop_before_pixels=True, force=True)
            return int(getattr(dcm, "InstanceNumber", 0))
        except Exception:
            return 0

    return sorted(files, key=sort_key)


def load_slice(fp):
    dcm = pydicom.dcmread(str(fp), force=True)
    arr = dcm.pixel_array

    while len(arr.shape) > 2 and arr.shape[0] == 1:
        arr = arr.squeeze(0)

    if len(arr.shape) == 3:
        if arr.shape[0] < arr.shape[-1]:
            arr = arr[arr.shape[0] // 2]
        else:
            arr = arr[:, :, arr.shape[-1] // 2]

    if len(arr.shape) != 2:
        raise ValueError(f"Unexpected slice shape: {arr.shape}")

    arr = arr.astype(np.float32)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)

    return arr


def extract_bbox(metadata):
    annotations = metadata.get("annotations", [])
    if not annotations:
        return None

    ann = annotations[0]
    bbox = ann.get("bounding_box_3d")
    if not isinstance(bbox, dict):
        return None

    x_min = float(bbox.get("start_column", 0))
    x_max = float(bbox.get("end_column", 0))
    y_min = float(bbox.get("start_row", 0))
    y_max = float(bbox.get("end_row", 0))

    z_start = bbox.get("start_slice")
    if z_start is None:
        z_start = bbox.get("start_image", 0)

    z_end = bbox.get("end_slice")
    if z_end is None:
        z_end = bbox.get("end_image", 0)

    try:
        z_min = int(float(z_start))
        z_max = int(float(z_end))
    except Exception:
        z_min, z_max = 0, 0

    if x_max <= x_min or y_max <= y_min:
        return None

    return {
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
    }


def clip_index(idx, n):
    if n <= 0:
        return 0
    return int(max(0, min(idx, n - 1)))


def draw_bbox(ax, img, bbox):
    h, w = img.shape

    # Annotation coordinates are commonly in 512x512 space.
    sx = w / 512.0
    sy = h / 512.0

    x = bbox["x_min"] * sx
    y = bbox["y_min"] * sy
    bw = (bbox["x_max"] - bbox["x_min"]) * sx
    bh = (bbox["y_max"] - bbox["y_min"]) * sy

    rect = patches.Rectangle(
        (x, y),
        bw,
        bh,
        linewidth=2,
        edgecolor="lime",
        facecolor="none",
    )
    ax.add_patch(rect)


def visualize_patient(patient_dir, output_dir):
    data_file = patient_dir / "patient_data.json"
    mri_dir = patient_dir / "MRI_DICOM_sample"

    if not data_file.exists() or not mri_dir.exists():
        return False, "Missing patient_data.json or MRI_DICOM_sample"

    metadata = load_json(data_file)
    bbox = extract_bbox(metadata)
    if bbox is None:
        return False, "No valid bounding_box_3d found"

    dicom_files = get_dicom_files(mri_dir)
    if not dicom_files:
        return False, "No DICOM files found"

    n = len(dicom_files)
    z0 = clip_index(bbox["z_min"], n)
    z1 = clip_index((bbox["z_min"] + bbox["z_max"]) // 2, n)
    z2 = clip_index(bbox["z_max"], n)
    pick = [z0, z1, z2]

    imgs = []
    for idx in pick:
        try:
            imgs.append((idx, load_slice(dicom_files[idx])))
        except Exception:
            return False, f"Failed to load slice index {idx}"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (idx, img), title in zip(axes, imgs, ["z_min", "z_mid", "z_max"]):
        ax.imshow(img, cmap="gray")
        draw_bbox(ax, img, bbox)
        ax.set_title(f"{title} (slice={idx})")
        ax.axis("off")

    patient_name = patient_dir.name
    fig.suptitle(
        f"{patient_name} | GT 3D BBox\n"
        f"x:[{bbox['x_min']:.1f},{bbox['x_max']:.1f}] "
        f"y:[{bbox['y_min']:.1f},{bbox['y_max']:.1f}] "
        f"z:[{bbox['z_min']},{bbox['z_max']}]",
        fontsize=12,
    )
    plt.tight_layout()

    out_file = output_dir / f"{patient_name}_gt_bbox.png"
    plt.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)

    return True, str(out_file)


def main():
    args = parse_args()

    base = Path(args.export_dir)
    split_dir = base / args.split
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    if not patient_dirs:
        raise RuntimeError(f"No patient folders found in: {split_dir}")

    print("=" * 70)
    print("GT BBOX MRI VISUALIZATION")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Total available patients: {len(patient_dirs)}")
    print(f"Requested: {args.num_patients}")
    print(f"Output dir: {output_dir}\n")

    count_ok = 0
    count_seen = 0

    for patient_dir in patient_dirs:
        if count_ok >= args.num_patients:
            break

        count_seen += 1
        ok, msg = visualize_patient(patient_dir, output_dir)
        if ok:
            count_ok += 1
            print(f"[{count_ok}/{args.num_patients}] {patient_dir.name}: saved -> {msg}")
        else:
            print(f"[skip] {patient_dir.name}: {msg}")

    print("\n" + "=" * 70)
    print(f"Done. Generated {count_ok} patient visualizations (scanned {count_seen} folders).")
    print("=" * 70)


if __name__ == "__main__":
    main()
