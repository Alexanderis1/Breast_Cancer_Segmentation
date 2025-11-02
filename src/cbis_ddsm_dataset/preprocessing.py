""" 
Brest Cancer Segmentation
----------------------------
Student Names:          Alessandro Anzellini - 2031062
                        Djibril Coulybaly - 2162985
                        Clara Gonçalves - 2128459 

Description of File:    Data processing script used to extract, organize, and export 
                        patient DICOM data (from the CBIS-DDSM dataset) into a structured 
                        format for training and testing.
"""

# Initialize imports
from pathlib import Path
from typing import Optional, Tuple, List
from PIL import Image
from tqdm import tqdm
import re
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import os

# Initizialize constant variables
ROOT     = Path(".").resolve()
DESC_DIR = ROOT / "description"

# Regex syntax for folder name of image file
# e.g. "Mass-Training_P_00112_RIGHT_MLO" or "Calc-Test_P_00038_LEFT_CC_1"
CASE_SEGMENT_RE = re.compile(
    r"^(Mass|Calc)-(Training|Test)_P_(\d{5})_(LEFT|RIGHT)_(CC|MLO)(?:_(\d+))?$",
    re.I
)

# non-breaking space
NBSP = "\u00A0"  

# Function to get the data split (train or test) from token
def _get_split(token: str) -> str:
    """Function to get the data split (train or test) from token."""
    return "train" if str(token).lower().startswith("train") else "test"

# Function to format the laterality to 'L' or 'R', akin to DICOM standards
def _format_laterality(v: str) -> str:
    """Function to format the laterality to 'L' or 'R', akin to DICOM standards."""
    v = (str(v) or "").upper().replace(NBSP, " ").strip().replace(" ", "")
    if v in ("L", "LEFT"):  return "L"
    if v in ("R", "RIGHT"): return "R"
    if v.startswith("L"):   return "L"
    if v.startswith("R"):   return "R"
    return v

# Function to format the image view to 'CC' (Cranio-Caudal) or 'MLO' (Medio-Lateral Oblique).
def _format_image_view(v: str) -> str:
    """Function to format the image view to 'CC' (Cranio-Caudal) or 'MLO' (Medio-Lateral Oblique)."""
    v = (str(v) or "").upper().replace(NBSP, " ").strip().replace(" ", "")
    if v in ("CC", "MLO"): return v
    if v.endswith("CC"):   return "CC"
    if v.endswith("MLO"):  return "MLO"
    return v
    
# Function to safely parse an integer
def _safe_int(x) -> Optional[int]:
    """Function to safely parse an integer."""
    try:
        return int(str(x).strip())
    except Exception:
        return None

# Function to format enum-like text (e.g shape, margins, type, and distribution) to upper case, and removing spaces, hyphens, underscores, and NBSP
def _format_enum(s: Optional[str]) -> Optional[str]:
    """Function to format enum-like text (e.g shape, margins, type, and distribution) to upper case, and removing spaces, hyphens, underscores, and NBSP"""
    if s is None: return None
    s = str(s).replace(NBSP, " ").strip()
    s = s.replace("-", " ").replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().upper()
    return s or None

# Function to format pathology to 'BENIGN' or 'MALIGNANT'
def _format_pathology(s: Optional[str]) -> Optional[str]:
    """Function to format pathology to 'BENIGN' or 'MALIGNANT'."""
    v = _format_enum(s)
    if not v: return None
    if v.startswith("BENIGN"):         return "BENIGN"
    if v in ("MALIGNANT", "MALIGNANCY"): return "MALIGNANT"
    return v

# Function to format the header text in the CSV headers (lower case and removing spaces, hyphens, underscores, and NBSP).
def _format_csv_columns(s: str) -> str:
    """Function to format the header text in the CSV headers (lower case and removing spaces, hyphens, underscores, and NBSP)."""
    return re.sub(r"[\s_\-]", "", str(s).lower().replace(NBSP, " "))

# Function to select the best matching header text from a list of possible texts (Exact match first, then fuzzy match).
def _match_csv_columns(cols, *cands) -> Optional[str]:
    """Function to select the best matching header text from a list of possible texts (Exact match first, then fuzzy match)."""
    norm = { _format_csv_columns(c): c for c in cols }
    for cand in cands:
        k = _format_csv_columns(cand)
        if k in norm: return norm[k]
    cand_norms = [_format_csv_columns(c) for c in cands]
    return next((c for c in cols if any(k in _format_csv_columns(c) for k in cand_norms)), None)

# Function to extract structured information (abnormality, split, patient_id, laterality, view_position) from a file path
def parse_case_from_path(p: Path) -> Optional[Tuple[str,str,str,str,str]]:
    """Function to extract structured information (abnormality, split, patient_id, laterality, view_position) from a file path"""
    for part in p.parts:
        m = CASE_SEGMENT_RE.match(part)
        if m:
            abn, split_tok, pid_num, lat, view, _ = m.groups()
            return ("calc" if abn.lower().startswith("calc") else "mass", _get_split(split_tok), f"P_{pid_num}", "L" if lat.upper()=="LEFT" else "R", view.upper())
    return None

# Function to extract rank/index (i.e. 0, 1, or 2) of multiple images of the same image view, using the file path.
def parse_rank_from_path(p: Path) -> int:
    """Function to extract rank/index (i.e. 0, 1, or 2) of multiple images of the same image view, using the file path."""
    for part in p.parts:
        m = CASE_SEGMENT_RE.match(part)
        if m:
            return int(m.group(6)) if m.group(6) else 0
    for part in p.parts:
        m = re.search(r"(?:^|[_-])(CC|MLO)_(\d{1,2})(?:[^0-9]|$)", part, flags=re.I)
        if m:
            return int(m.group(2))
    return 0

# Function to format SeriesDescription text (lower case, remove non-alphanumeric characters and extra spaces)
def _format_text(s: str) -> str:
    """Function to format SeriesDescription text (lower case, remove non-alphanumeric characters and extra spaces)"""
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return s.strip()

# Function to extract the SeriesDescription from a DICOM file
def series_from_dcm(dcm_path: Path) -> Optional[str]:
    """Function to extract the SeriesDescription from a DICOM file"""
    try:
        ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True, force=True)
        return str(ds.get("SeriesDescription") or ds.get((0x0008, 0x103E)) or None)
    except Exception:
        return None

# Function to extract the file type (ROI Mask, Cropped, or Full Mammogram) from the SeriesDescription text
def _series_to_filetype_from_text(text: Optional[str]) -> Optional[str]:
    """Function to extract the file type (ROI Mask, Cropped, or Full Mammogram) from the SeriesDescription text"""
    s = _format_text(text or "")
    if not s: return None
    if ("roi" in s and "mask" in s) or "roi mask" in s or "mask images" in s or "roimask" in s: return "mask"
    if "cropped" in s or "crop" in s: return "cropped"
    if ("full" in s and "mammogram" in s) or "full mammogram images" in s or "mammogram full" in s: return "full"
    if "mask" in s: return "mask"
    return None

# Function to extract the file type from the parent folder names of the DICOM file path
def _series_to_filetype_from_path(dcm_path: Path) -> Optional[str]:
    """Function to extract the file type from the parent folder names of the DICOM file path"""
    p = dcm_path.parent
    names = [p.name for _ in range(4) if p and (p := p.parent)]
    return _series_to_filetype_from_text(_format_text(" / ".join(names)))

# Function to extract the file type either from the DICOM SeriesDescription, or the parent folder names
def series_to_filetype(series_desc: Optional[str], dcm_path: Optional[Path]) -> Optional[str]:
    """Function to extract the file type either from the DICOM SeriesDescription, or the parent folder names"""
    return _series_to_filetype_from_text(series_desc) or (_series_to_filetype_from_path(dcm_path) if dcm_path else None)

# Function to read CSV files into a Dataframe
def _read_csv_robust(path: Path) -> Optional[pd.DataFrame]:
    """Function to read CSV files into a Dataframe"""
    for enc in ("utf-8", "utf-8-sig", "latin1"):
        try:
            df = pd.read_csv(path, dtype=str, encoding=enc)
            df.columns = [str(c).replace(NBSP, " ").strip() for c in df.columns]
            return df
        except Exception:
            pass
    return None

# Function to find the manifest-* directory which contains the downloaded CBIS-DDSM data
def find_manifest_dir() -> Path:
    """Function to find the manifest-* directory which contains the downloaded CBIS-DDSM data"""
    return max(ROOT.glob("manifest-*"), key=lambda p: p.stat().st_mtime)

# Function to obtain the path of metadata.csv inside the manifest-* directory
def metadata_csv_path(manifest_dir: Path) -> Path:
    """Function to obtain the path of metadata.csv inside the manifest-* directory"""
    p = manifest_dir / "metadata.csv"
    if not p.exists(): raise FileNotFoundError(f"metadata.csv not found in {manifest_dir}")
    return p

# Function to obtain the File Location column name in metadata.csv
def _file_location_path(df: pd.DataFrame) -> str:
    """Function to obtain the File Location column name in metadata.csv"""
    return "File Location" if "File Location" in df.columns else _match_csv_columns(df.columns, "File Location")

# Function to gather all the DICOM file paths from the File Location column
def dcm_paths_from_file_location(manifest_dir: Path, rel: str) -> List[Path]:
    """Function to gather all the DICOM file paths from the File Location column"""
    p = manifest_dir / Path((rel or "").replace(NBSP, " ").strip().lstrip("\\/"))
    if p.is_file(): return [p] if p.suffix.lower() == ".dcm" else []
    return list(p.rglob("*.dcm")) if p.is_dir() else []

# Function that reads all the case description CSV files, preprocesses them and merges them a single DataFrame. 
# This is then used to buiid a tuple of label indexes, easily accessible by (patient_id, laterality, view_position).
def build_label_indexes(desc_dir: Path):
    """Function that reads all the case description CSV files, preprocesses them and merges them a single DataFrame. 
    This is then used to buiid a tuple of label indexes easily accessible by (patient_id, laterality, view_position)."""

    column_names = {
        'patient_id': ["patient_id"],
        'laterality': ["left or right breast", "left_right", "side"],
        'view_position': ["image view", "view"],
        'abnormality': ["abnormality type"],
        'breast_density': ["breast density", "breast_density"],
        'assessment': ["assessment"],
        'pathology': ["pathology"],
        'subtlety': ["subtlety"],
        'calc_type': ["calc type", "calcification type", "type of calcification"],
        'calc_distribution': ["calc distribution", "calcification distribution", "distribution"],
        'mass_shape': ["mass shape"],
        'mass_margins': ["mass margins", "mass margin"],
    }

    # Reading and merging all CSV files
    dfs = []
    for csv_file in desc_dir.glob("*case_description*_*_set*.csv"):
        df = _read_csv_robust(csv_file)
        if df is not None:
            rename_map = {}
            for std, cands in column_names.items():
                col = _match_csv_columns(df.columns, *cands)
                if col:
                    rename_map[col] = std
            df = df.rename(columns=rename_map)
            dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)

    # Formatting merged dataframe and filtering valid entries
    colmap = {k: k for k in column_names}
    df = df.dropna(subset=[colmap['patient_id'], colmap['laterality'], colmap['view_position']])
    df['pid'] = df[colmap['patient_id']].astype(str).str.strip()
    df['lat'] = df[colmap['laterality']].map(_format_laterality)
    df['view'] = df[colmap['view_position']].map(_format_image_view)
    df = df[df['lat'].isin(['L','R']) & df['view'].isin(['CC','MLO'])]
    k3 = ['pid','lat','view']

    # Function that finds the most common value in a list, ignoring any missing values
    def mode1(x):
        """Function that finds the most common value in a list, ignoring any missing values."""
        x = pd.Series([v for v in x if pd.notnull(v)])
        return x.mode().iloc[0] if not x.mode().empty else None

    # Building label indexes grouped by (patient_id, laterality, view_position)
    
    # Breast density index
    density = df.groupby(k3)[colmap['breast_density']].agg(lambda x: mode1([_safe_int(v) for v in x])).to_dict()

    # General labels index found in both calc and mass cases
    general = df.groupby(k3).agg({
        colmap['assessment']: lambda x: mode1([_safe_int(v) for v in x]),
        colmap['pathology']: lambda x: mode1([_format_pathology(v) for v in x]),
        colmap['subtlety']: lambda x: mode1([_safe_int(v) for v in x]),
    }).rename(columns={
        colmap['assessment']: 'assessment',
        colmap['pathology']: 'pathology',
        colmap['subtlety']: 'subtlety',
    }).to_dict(orient='index')

    # Labels specific to calc cases
    calc_only = df.groupby(k3).agg({
        colmap['calc_type']: lambda x: mode1([_format_enum(v) for v in x]),
        colmap['calc_distribution']: lambda x: mode1([_format_enum(v) for v in x]),
    }).rename(columns={
        colmap['calc_type']: 'calc_type',
        colmap['calc_distribution']: 'calc_distribution',
    }).to_dict(orient='index')

    # Labels specific to mass cases
    mass_only = df.groupby(k3).agg({
        colmap['mass_shape']: lambda x: mode1([_format_enum(v) for v in x]),
        colmap['mass_margins']: lambda x: mode1([_format_enum(v) for v in x]),
    }).rename(columns={
        colmap['mass_shape']: 'mass_shape',
        colmap['mass_margins']: 'mass_margins',
    }).to_dict(orient='index')

    # Convert label indexes to tuple keys
    density = {tuple(k): v for k, v in density.items()}
    general = {tuple(k): v for k, v in general.items()}
    calc_only = {tuple(k): v for k, v in calc_only.items()}
    mass_only = {tuple(k): v for k, v in mass_only.items()}

    return density, general, calc_only, mass_only

# Function to create a DataFrame that preprocesses and merges the metadata and case descriptions from the CBIS-DDSM dataset
def build_cbis_dataframe():
    """Function to create a DataFrame that preprocesses and merges the metadata and case descriptions from the CBIS-DDSM dataset"""
    columns = [
        'patient_id', 'laterality', 'view_position', 'abnormality', 'split', 'rank',
        'breast_density', 'assessment', 'pathology', 'subtlety',
        'calc_type', 'calc_distribution', 'mass_shape', 'mass_margins',
        'file_type', 'file_path', 'source', 'inserted_at'
    ]

    # Build label indexes from case description CSV files
    density, general, calc_only, mass_only = build_label_indexes(DESC_DIR)

    # Parsing metadata.csv to extract DICOM file paths and case information
    manifest_dir = find_manifest_dir()
    meta_path = metadata_csv_path(manifest_dir)
    meta = _read_csv_robust(meta_path)
    if meta is None:
        print(f"[error] Unable to read metadata.csv at {meta_path}")
        return None
    file_loc_col = _file_location_path(meta)

    # Using the metadata and the label indexes for each DICOM file to build the final DataFrame
    rows = []
    seen = set()
    skipped_case = 0
    skipped_series = 0

    for _, row in meta.iterrows():
        rel = (row.get(file_loc_col) or "").strip()
        if not rel:
            continue
        for dcm_path in dcm_paths_from_file_location(manifest_dir, rel):
            spath = str(dcm_path)
            if spath in seen:
                continue
            seen.add(spath)

            parsed = parse_case_from_path(dcm_path)
            if not parsed:
                skipped_case += 1
                continue
            abn, split, pid, lat, view = parsed

            sdesc = series_from_dcm(dcm_path)
            ftype = series_to_filetype(sdesc, dcm_path)
            if not ftype:
                skipped_series += 1
                continue

            rank = parse_rank_from_path(dcm_path)

            # labels from indexes (MATCH ONLY BY: (pid, lat, view))
            k3 = (pid, lat, view)
            bd = density.get(k3)
            g  = general.get(k3, {})
            if abn == "calc":
                c = calc_only.get(k3, {})
                m = {}
            else:
                c = {}
                m = mass_only.get(k3, {})

            row_dict = {
                'patient_id': pid,
                'laterality': lat,
                'view_position': view,
                'abnormality': abn,
                'split': split,
                'rank': rank,
                'breast_density': bd,
                'assessment': g.get('assessment'),
                'pathology': g.get('pathology'),
                'subtlety': g.get('subtlety'),
                'calc_type': c.get('calc_type'),
                'calc_distribution': c.get('calc_distribution'),
                'mass_shape': m.get('mass_shape'),
                'mass_margins': m.get('mass_margins'),
                'file_type': ftype,
                'file_path': spath,
                'source': 'CBIS-DDSM',
                'inserted_at': pd.Timestamp.now(),
            }
            rows.append(row_dict)

    #  Create DataFrame
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv('merged_database.csv', index=False)

    # Print summary
    print("The DataFrame has been saved as merged_database.csv")
    print(f"The DataFrame has {len(df)} rows in total")
    print(f"{skipped_case} have been skipped (no case segment): ")
    print(f"{skipped_series} have been skipped (unknown SeriesDescription): ")
    if not df.empty:
        print(f"The DataFrame has the following rows for each file type: {df['file_type'].value_counts().to_dict()}")
        print(f"The DataFrame has the following rows for each split: {df['split'].value_counts().to_dict()}")

    return df

# Function to display sample images for each file type in the newly merged database
def plot_sample_images(df, dataset_name):
    """
    Function to display sample images for each file type in the newly merged database
    """
    plt.figure(figsize=(15, 5))
    types = [
        ('full', 'Full Mammogram'),
        ('cropped', 'Cropped'),
        ('mask', 'ROI Mask')
    ]
    for idx, (ftype, title) in enumerate(types):
        subset = df[df['file_type'] == ftype]
        if not subset.empty:
            sample_path = subset['file_path'].dropna().iloc[0]
            try:
                ds = pydicom.dcmread(sample_path)
                img = ds.pixel_array
                plt.subplot(1, 3, idx + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"{dataset_name} - {title}")
                plt.axis('off')
            except Exception as e:
                print(f"Could not load DICOM for {title}. Path: {sample_path}. Error: {e}")
        else:
            print(f"No valid paths found for {title}.")
    plt.tight_layout()
    plt.show()

# Main function
def main():
    print("*" * 25)
    print("Preprocessing CBIS-DDSM Dataset")
    print("*" * 25)

    print("1) Check if dataset has been preprocessed or not")
    master_db_path = ROOT / 'merged_database.csv'
    if master_db_path.exists():
        print("The database has already exists. Skipping build.")
        merged_database = pd.read_csv(master_db_path)
    else:
        print("The database is not processed. Beginning preprocessing...")
        merged_database = build_cbis_dataframe()
        print(f"Master database built with {len(merged_database)} entries.")
        merged_database.to_csv("merged_database.csv", index=False)

    print("\n2) Plot sample images from each file type from the preprocessed dataset")
    plot_sample_images(merged_database, "CBIS-DDSM")
    print("Please close the image window to continue...")

    print("\n3) Preprocess and save DICOM images as .npy files for quicker loading during training/testing")
    output_dir = "preprocessed_npy"
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_paths = []

    for idx, row in tqdm(merged_database.iterrows(), total=len(merged_database), desc="Preprocessing DICOMs"):
        dcm_path = row['file_path']
        try:
            dcm = pydicom.dcmread(dcm_path)
            img = dcm.pixel_array

            # Normalize to 0-255
            if img.min() == img.max():
                img = np.zeros_like(img)
            else:
                img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
            
            # Resize to 224x224 (if needed)
            img_pil = Image.fromarray(img)
            img_pil = img_pil.resize((224, 224))
            img_arr = np.array(img_pil)
            
            # If grayscale, add channel dimension (224, 224, 3)
            if img_arr.ndim == 2:
                img_arr = np.stack([img_arr]*3, axis=-1)

            # Save as .npy
            out_path = os.path.join(output_dir, f"img_{idx}.npy")
            np.save(out_path, img_arr)
            preprocessed_paths.append(out_path)
        except Exception as e:
            print(f"Error processing {dcm_path}: {e}")
            preprocessed_paths.append("")

    # Add to CSV and save
    merged_database['preprocessed_path'] = preprocessed_paths
    merged_database.to_csv("final_database.csv", index=False)
    print(f"Completed preprocessing of database. The final database CSV is saved as 'final_database.csv' and includes the file path to the DICOM images saved as .npy files in '{output_dir}'.")

if __name__ == "__main__":
    main()