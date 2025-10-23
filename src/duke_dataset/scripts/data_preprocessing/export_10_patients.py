import pandas as pd
import os
import random
import json
import shutil
from pathlib import Path

# Dataset paths
DATA_DIR = '/mnt/c/Users/35193/Desktop/duke_dataset/data'
SEGMENTATION_DIR_V2 = '/mnt/c/Users/35193/Desktop/duke_dataset/data/PKG - Duke-Breast-Cancer-MRI-Supplement-v2/Duke-Breast-Cancer-MRI-Supplement-v2/Segmentation_Masks_NRRD'
SEGMENTATION_DIR_V3 = '/mnt/c/Users/35193/Desktop/duke_dataset/data/PKG - Duke-Breast-Cancer-MRI-Supplement-v3/Duke-Breast-Cancer-MRI-Supplement-v3/Segmentation_Masks_NRRD'
OUT_DIR = os.getcwd()
EXPORT_DIR = os.path.join(OUT_DIR, 'exported_10_patients')
NUM_TRAIN_PATIENTS = 20
NUM_TEST_PATIENTS = 5

def load_all_data():
    """Load all Excel and CSV files with proper error handling"""
    data = {}
    
    print(f"Loading data files from: {DATA_DIR}\n")
    
    if not os.path.exists(DATA_DIR):
        print(f"ERROR: DATA_DIR does not exist: {DATA_DIR}")
        return {}
    
    # Excel files - Clinical features have TWO header rows
    # Row 0: Category headers (Patient Information, MRI Technical Information, etc.)
    # Row 1: Actual column names (Patient ID, Days to MRI, etc.)
    excel_files = {
        'annotation_boxes': {
            'file': 'Annotation_Boxes.xlsx',
            'header_row': 0,  
            'expected_cols': ['Patient ID', 'Start Row', 'End Row', 'Start Column', 'End Column', 'Start Slice', 'End Slice']
        },
        'density_assessments': {
            'file': 'Breast_Radiologist_Density_Assessments.xlsx',
            'header_row': 0, 
            'expected_cols': ['Subject_ID', 'Radiologist A', 'Radiologist B', 'Radiologist C']
        },
        'filepath_mapping': {
            'file': 'Breast-Cancer-MRI-filepath_filename-mapping.xlsx',
            'header_row': 0, 
            'expected_cols': ['sop_instance_UID', 'original_path_and_filename']
        },
        'clinical_features': {
            'file': 'Clinical_and_Other_Features.xlsx',
            'header_row': 1,  # USE SECOND ROW (index 1) as column names
            'expected_cols': ['Patient ID']
        },
        'imaging_features': {
            'file': 'Imaging_Features.xlsx',
            'header_row': 0,  
            'expected_cols': ['Patient ID']
        }
    }
    
    for key, config in excel_files.items():
        file_path = os.path.join(DATA_DIR, config['file'])
        if os.path.exists(file_path):
            try:
                # Use the specified header row from config
                df = pd.read_excel(file_path, header=config['header_row'])
                data[key] = df
                print(f"✓ Loaded {key}: {len(df)} rows, {len(df.columns)} columns")
                print(f"  Using header row: {config['header_row']}")
                
                # Verify expected columns
                missing_cols = [col for col in config['expected_cols'] if col not in df.columns]
                if missing_cols:
                    print(f"  WARNING: Missing expected columns: {missing_cols}")
                
                # Show first few column names for verification
                print(f"  Columns: {df.columns.tolist()[:5]}...")
                
                # Show sample patient IDs if Patient ID column exists
                if 'Patient ID' in df.columns:
                    sample_ids = df['Patient ID'].dropna().head(3).tolist()
                    print(f"  Sample Patient IDs: {sample_ids}")
                elif 'Subject_ID' in df.columns:
                    sample_ids = df['Subject_ID'].dropna().head(3).tolist()
                    print(f"  Sample Subject IDs: {sample_ids}")
                    
            except Exception as e:
                print(f"✗ Error loading {config['file']}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"✗ File not found: {config['file']}")
            data[key] = pd.DataFrame()
    
    # CSV files
    csv_files = {
        'segmentation_mapping': 'segmentation_filepath_mapping.csv',
        'train_ids': 'train_ids.csv',
        'test_ids': 'test_ids.csv'
    }
    
    for key, filename in csv_files.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                data[key] = df
                print(f"✓ Loaded {key}: {len(df)} rows")
                print(f"  Columns: {df.columns.tolist()}")
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"✗ File not found: {filename}")
            data[key] = pd.DataFrame()
    
    return data

def get_available_patients():
    """Get list of all available patient IDs from directory structure"""
    patients = set()
    
    try:
        manifest_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith('manifest-')]
        
        for manifest_dir in manifest_dirs:
            duke_mri_path = os.path.join(DATA_DIR, manifest_dir, 'Duke-Breast-Cancer-MRI')
            if os.path.exists(duke_mri_path):
                for folder in os.listdir(duke_mri_path):
                    if folder.startswith('Breast_MRI_') and os.path.isdir(os.path.join(duke_mri_path, folder)):
                        # Extract patient number (e.g., "001" from "Breast_MRI_001")
                        patient_num = folder.replace('Breast_MRI_', '')
                        patients.add(patient_num)
        
        print(f"\nFound {len(patients)} patients from MRI directories")
        if patients:
            print(f"Sample patient IDs: {sorted(list(patients))[:5]}")
        
    except Exception as e:
        print(f"Error finding patients: {e}")
    
    return sorted(list(patients))

def normalize_patient_id(patient_id):
    """Normalize patient ID to standard format with leading zeros"""
    # Remove "Breast_MRI_" prefix if present
    if isinstance(patient_id, str) and patient_id.startswith('Breast_MRI_'):
        patient_id = patient_id.replace('Breast_MRI_', '')
    
    # Convert to string and pad with zeros
    patient_num = str(patient_id).zfill(3)
    return patient_num

def format_patient_id_variations(patient_id):
    """Generate all possible patient ID formats for matching"""
    normalized = normalize_patient_id(patient_id)
    
    return {
        'normalized': normalized,  # "001"
        'full_format': f"Breast_MRI_{normalized}",  # "Breast_MRI_001"
        'no_leading_zeros': str(int(normalized)),  # "1"
        'full_no_zeros': f"Breast_MRI_{int(normalized)}"  # "Breast_MRI_1"
    }

def extract_clinical_features(patient_id, df):
    """Extract clinical features for a patient"""
    if df.empty or 'Patient ID' not in df.columns:
        return {}
    
    variations = format_patient_id_variations(patient_id)
    
    # Try matching with different ID formats
    for var_name, var_value in variations.items():
        mask = df['Patient ID'] == var_value
        if mask.any():
            data = df[mask].iloc[0].to_dict()
            # Remove NaN values
            return {k: v for k, v in data.items() if pd.notna(v)}
    
    return {}

def extract_imaging_features(patient_id, df):
    """Extract imaging features for a patient"""
    if df.empty or 'Patient ID' not in df.columns:
        return {}
    
    variations = format_patient_id_variations(patient_id)
    
    for var_name, var_value in variations.items():
        mask = df['Patient ID'] == var_value
        if mask.any():
            data = df[mask].iloc[0].to_dict()
            return {k: v for k, v in data.items() if pd.notna(v)}
    
    return {}

def extract_annotations(patient_id, df):
    """Extract 3D bounding box annotations for a patient"""
    annotations = []
    
    if df.empty or 'Patient ID' not in df.columns:
        return annotations
    
    variations = format_patient_id_variations(patient_id)
    
    # Try matching with full format first (most likely)
    mask = df['Patient ID'] == variations['full_format']
    
    if not mask.any():
        # Try other variations
        for var_value in variations.values():
            mask = df['Patient ID'] == var_value
            if mask.any():
                break
    
    if not mask.any():
        return annotations
    
    matched_rows = df[mask]
    
    # Required columns for 3D bounding box
    required_cols = ['Start Row', 'End Row', 'Start Column', 'End Column', 'Start Slice', 'End Slice']
    
    if not all(col in df.columns for col in required_cols):
        print(f"    WARNING: Missing required annotation columns")
        return annotations
    
    for _, row in matched_rows.iterrows():
        try:
            bbox_3d = {
                'start_row': int(row['Start Row']),
                'end_row': int(row['End Row']),
                'start_column': int(row['Start Column']),
                'end_column': int(row['End Column']),
                'start_slice': int(row['Start Slice']),
                'end_slice': int(row['End Slice'])
            }
            
            # Calculate dimensions
            bbox_3d['height'] = bbox_3d['end_row'] - bbox_3d['start_row'] + 1
            bbox_3d['width'] = bbox_3d['end_column'] - bbox_3d['start_column'] + 1
            bbox_3d['depth'] = bbox_3d['end_slice'] - bbox_3d['start_slice'] + 1
            bbox_3d['volume'] = bbox_3d['height'] * bbox_3d['width'] * bbox_3d['depth']
            
            annotation = {
                'patient_id': variations['full_format'],
                'bounding_box_3d': bbox_3d,
                'annotation_type': '3D_Bounding_Box'
            }
            
            annotations.append(annotation)
            
        except Exception as e:
            print(f"    Warning: Error processing annotation row: {e}")
            continue
    
    return annotations

def extract_density_assessment(patient_id, df):
    """Extract density assessment for a patient"""
    if df.empty or 'Subject_ID' not in df.columns:
        return {}
    
    variations = format_patient_id_variations(patient_id)
    
    for var_value in variations.values():
        mask = df['Subject_ID'] == var_value
        if mask.any():
            data = df[mask].iloc[0].to_dict()
            return {k: v for k, v in data.items() if pd.notna(v)}
    
    return {}

def extract_mri_file_mappings(patient_id, df):
    """Extract MRI file path mappings for a patient"""
    if df.empty or 'original_path_and_filename' not in df.columns:
        return []
    
    variations = format_patient_id_variations(patient_id)
    
    # Filter rows where path contains patient ID
    mask = df['original_path_and_filename'].str.contains(
        variations['full_format'], 
        na=False, 
        case=False
    )
    
    if mask.any():
        mri_files = df[mask].to_dict('records')
        return [{k: v for k, v in mri.items() if pd.notna(v)} for mri in mri_files]
    
    return []

def extract_segmentation_files(patient_id, df):
    """Extract segmentation file mappings for a patient"""
    if df.empty:
        return [], []
    
    # Handle potential column name typo (atient ID vs Patient ID)
    patient_col = None
    for col in df.columns:
        if 'patient' in col.lower() and 'id' in col.lower():
            patient_col = col
            break
    
    if patient_col is None:
        return [], []
    
    variations = format_patient_id_variations(patient_id)
    
    # Match using full format (most common in segmentation file)
    mask = df[patient_col] == variations['full_format']
    
    if mask.any():
        seg_data = df[mask]
        seg_files = seg_data.to_dict('records')
        seg_labels = seg_data['Segmentation Label'].unique().tolist() if 'Segmentation Label' in seg_data.columns else []
        return seg_files, seg_labels
    
    return [], []

def extract_nrrd_segmentation_masks(patient_id):
    """Extract NRRD segmentation masks from both v2 and v3 data structures"""
    normalized_id = normalize_patient_id(patient_id)
    variations = format_patient_id_variations(normalized_id)
    
    segmentation_data = {
        'v2_masks': {'breast_masks': {'train': [], 'test': []}, 'dense_tissue_masks': {'train': [], 'test': []}},
        'v3_masks': {'breast_masks': [], 'dense_and_vessels_masks': []},
        'found_masks': False,
        'data_versions': []
    }
    
    # === V2 Data Structure ===
    if os.path.exists(SEGMENTATION_DIR_V2):
        v2_categories = {
            'breast_train': 'breast_masks.train',
            'breast_test': 'breast_masks.test', 
            'dense_tissue_train': 'dense_tissue_masks.train',
            'dense_tissue_test': 'dense_tissue_masks.test'
        }
        
        v2_found = False
        for category, data_path in v2_categories.items():
            category_dir = os.path.join(SEGMENTATION_DIR_V2, category)
            if not os.path.exists(category_dir):
                continue
                
            for filename in os.listdir(category_dir):
                if filename.lower().endswith('.nrrd'):
                    # For V2, use more precise matching to avoid false positives
                    # V2 files follow pattern: Breast_MRI_XXX_pre_YYY.nrrd
                    for var_value in variations.values():
                        # Check if filename contains the exact patient format
                        if f"breast_mri_{var_value.lower()}_pre_" in filename.lower():
                            file_path = os.path.join(category_dir, filename)
                            file_info = {
                                'filename': filename,
                                'full_path': file_path,
                                'category': category,
                                'data_version': 'v2',
                                'file_size': os.path.getsize(file_path) if os.path.exists(file_path) else 0
                            }
                            
                            if 'breast_masks' in data_path:
                                if 'train' in data_path:
                                    segmentation_data['v2_masks']['breast_masks']['train'].append(file_info)
                                else:
                                    segmentation_data['v2_masks']['breast_masks']['test'].append(file_info)
                            else:
                                if 'train' in data_path:
                                    segmentation_data['v2_masks']['dense_tissue_masks']['train'].append(file_info)
                                else:
                                    segmentation_data['v2_masks']['dense_tissue_masks']['test'].append(file_info)
                            
                            v2_found = True
                            break
        
        if v2_found:
            segmentation_data['data_versions'].append('v2')
            segmentation_data['found_masks'] = True
    
    # === V3 Data Structure ===
    if os.path.exists(SEGMENTATION_DIR_V3):
        patient_dir_v3 = os.path.join(SEGMENTATION_DIR_V3, variations['full_format'])
        
        if os.path.exists(patient_dir_v3):
            v3_found = False
            
            # Look for breast mask
            breast_mask_pattern = f"Segmentation_{variations['full_format']}_Breast.seg.nrrd"
            breast_mask_path = os.path.join(patient_dir_v3, breast_mask_pattern)
            
            if os.path.exists(breast_mask_path):
                file_info = {
                    'filename': breast_mask_pattern,
                    'full_path': breast_mask_path,
                    'mask_type': 'breast',
                    'data_version': 'v3',
                    'file_size': os.path.getsize(breast_mask_path)
                }
                segmentation_data['v3_masks']['breast_masks'].append(file_info)
                v3_found = True
            
            # Look for dense tissue and vessels mask
            dense_vessels_mask_pattern = f"Segmentation_{variations['full_format']}_Dense_and_Vessels.seg.nrrd"
            dense_vessels_mask_path = os.path.join(patient_dir_v3, dense_vessels_mask_pattern)
            
            if os.path.exists(dense_vessels_mask_path):
                file_info = {
                    'filename': dense_vessels_mask_pattern,
                    'full_path': dense_vessels_mask_path,
                    'mask_type': 'dense_and_vessels',
                    'data_version': 'v3',
                    'file_size': os.path.getsize(dense_vessels_mask_path)
                }
                segmentation_data['v3_masks']['dense_and_vessels_masks'].append(file_info)
                v3_found = True
            
            if v3_found:
                segmentation_data['data_versions'].append('v3')
                segmentation_data['found_masks'] = True
    
    # Calculate summary statistics
    v2_total = (len(segmentation_data['v2_masks']['breast_masks']['train']) + 
               len(segmentation_data['v2_masks']['breast_masks']['test']) +
               len(segmentation_data['v2_masks']['dense_tissue_masks']['train']) + 
               len(segmentation_data['v2_masks']['dense_tissue_masks']['test']))
    
    v3_total = (len(segmentation_data['v3_masks']['breast_masks']) + 
               len(segmentation_data['v3_masks']['dense_and_vessels_masks']))
    
    segmentation_data['summary'] = {
        'total_nrrd_masks': v2_total + v3_total,
        'v2_masks_count': v2_total,
        'v3_masks_count': v3_total,
        'data_versions_found': segmentation_data['data_versions'],
        'has_breast_masks': (len(segmentation_data['v2_masks']['breast_masks']['train']) + 
                            len(segmentation_data['v2_masks']['breast_masks']['test']) + 
                            len(segmentation_data['v3_masks']['breast_masks'])) > 0,
        'has_dense_tissue_masks': (len(segmentation_data['v2_masks']['dense_tissue_masks']['train']) + 
                                  len(segmentation_data['v2_masks']['dense_tissue_masks']['test']) + 
                                  len(segmentation_data['v3_masks']['dense_and_vessels_masks'])) > 0,
        'mask_types': []
    }
    
    if segmentation_data['summary']['has_breast_masks']:
        segmentation_data['summary']['mask_types'].append('breast')
    if segmentation_data['summary']['has_dense_tissue_masks']:
        segmentation_data['summary']['mask_types'].append('dense_tissue_vessels')
    
    return segmentation_data

def check_patient_has_segmentation_masks(patient_id):
    """Quick check if patient has any segmentation masks available"""
    nrrd_data = extract_nrrd_segmentation_masks(patient_id)
    return nrrd_data['found_masks']

def get_dataset_split(patient_id, train_df, test_df):
    """Determine if patient is in train or test split"""
    variations = format_patient_id_variations(patient_id)
    
    # Check train set
    if not train_df.empty:
        for col in train_df.columns:
            for var_value in variations.values():
                if var_value in train_df[col].values or str(var_value) in train_df[col].astype(str).values:
                    return 'train'
    
    # Check test set
    if not test_df.empty:
        for col in test_df.columns:
            for var_value in variations.values():
                if var_value in test_df[col].values or str(var_value) in test_df[col].astype(str).values:
                    return 'test'
    
    return 'unknown'

def extract_patient_data(patient_id, all_data):
    """Extract all data for a specific patient"""
    normalized_id = normalize_patient_id(patient_id)
    variations = format_patient_id_variations(normalized_id)
    
    print(f"  Extracting data for patient {normalized_id} ({variations['full_format']})...")
    
    # Extract all data components
    clinical = extract_clinical_features(normalized_id, all_data['clinical_features'])
    imaging = extract_imaging_features(normalized_id, all_data['imaging_features'])
    annotations = extract_annotations(normalized_id, all_data['annotation_boxes'])
    density = extract_density_assessment(normalized_id, all_data['density_assessments'])
    mri_files = extract_mri_file_mappings(normalized_id, all_data['filepath_mapping'])
    seg_files, seg_labels = extract_segmentation_files(normalized_id, all_data['segmentation_mapping'])
    nrrd_segmentation = extract_nrrd_segmentation_masks(normalized_id)
    split = get_dataset_split(normalized_id, all_data['train_ids'], all_data['test_ids'])
    
    # Create annotation summary
    annotation_summary = {
        'total_annotations': len(annotations),
        'has_3d_bounding_boxes': len(annotations) > 0,
        'annotation_types': ['3D_Bounding_Box'] if annotations else []
    }
    
    if annotations:
        # Calculate aggregate statistics
        volumes = [ann['bounding_box_3d']['volume'] for ann in annotations]
        annotation_summary['total_volume'] = sum(volumes)
        annotation_summary['avg_volume'] = sum(volumes) / len(volumes)
        annotation_summary['min_volume'] = min(volumes)
        annotation_summary['max_volume'] = max(volumes)
    
    patient_data = {
        'patient_id': normalized_id,
        'patient_id_full': variations['full_format'],
        'demographic_clinical': clinical,
        'imaging_features': imaging,
        'annotations': annotations,
        'annotation_summary': annotation_summary,
        'density_assessment': density,
        'mri_files': mri_files,
        'segmentation_files': seg_files,
        'segmentation_labels': seg_labels,
        'nrrd_segmentation_masks': nrrd_segmentation,
        'dataset_split': split
    }
    
    # Print summary
    print(f"    Clinical data: {'✓' if clinical else '✗'} ({len(clinical)} fields)")
    print(f"    Imaging features: {'✓' if imaging else '✗'} ({len(imaging)} features)")
    print(f"    Annotations: {'✓' if annotations else '✗'} ({len(annotations)} 3D boxes)")
    if annotations:
        print(f"      Total annotated volume: {annotation_summary['total_volume']} voxels")
    print(f"    Density assessment: {'✓' if density else '✗'} ({len(density)} fields)")
    print(f"    MRI file mappings: {'✓' if mri_files else '✗'} ({len(mri_files)} files)")
    print(f"    Segmentation files (CSV): {'✓' if seg_files else '✗'} ({len(seg_files)} files)")
    if seg_labels:
        print(f"      Labels: {seg_labels}")
    nrrd_total = nrrd_segmentation['summary']['total_nrrd_masks'] if nrrd_segmentation['found_masks'] else 0
    nrrd_versions = ', '.join(nrrd_segmentation['summary']['data_versions_found']) if nrrd_segmentation['found_masks'] else ''
    print(f"    NRRD Segmentation masks: {'✓' if nrrd_segmentation['found_masks'] else '✗'} ({nrrd_total} masks)")
    if nrrd_segmentation['found_masks']:
        print(f"      Data versions: {nrrd_versions}")
        print(f"      Mask types: {', '.join(nrrd_segmentation['summary']['mask_types'])}")
        if nrrd_segmentation['summary']['v2_masks_count'] > 0:
            print(f"      v2 masks: {nrrd_segmentation['summary']['v2_masks_count']}")
        if nrrd_segmentation['summary']['v3_masks_count'] > 0:
            print(f"      v3 masks: {nrrd_segmentation['summary']['v3_masks_count']}")
    print(f"    Dataset split: {split}")
    
    return patient_data

def copy_mri_files(patient_id, export_patient_dir, max_files=20):
    """Copy sample MRI DICOM files for the patient"""
    mri_export_dir = os.path.join(export_patient_dir, 'MRI_DICOM_sample')
    os.makedirs(mri_export_dir, exist_ok=True)
    
    normalized_id = normalize_patient_id(patient_id)
    variations = format_patient_id_variations(normalized_id)
    
    copied_count = 0
    
    try:
        manifest_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith('manifest-')]
        
        for manifest_dir in manifest_dirs:
            patient_mri_dir = os.path.join(
                DATA_DIR, 
                manifest_dir, 
                'Duke-Breast-Cancer-MRI', 
                variations['full_format']
            )
            
            if os.path.exists(patient_mri_dir):
                # Get all DICOM files
                dcm_files = []
                for root, dirs, files in os.walk(patient_mri_dir):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            dcm_files.append(os.path.join(root, file))
                
                # Copy sample
                sample_files = dcm_files[:max_files]
                
                for src_path in sample_files:
                    try:
                        rel_path = os.path.relpath(src_path, patient_mri_dir)
                        dst_path = os.path.join(mri_export_dir, rel_path)
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"      Failed to copy {os.path.basename(src_path)}: {e}")
                
                # Save file inventory
                inventory = {
                    'total_dcm_files': len(dcm_files),
                    'copied_files': copied_count,
                    'sample_ratio': f"{copied_count}/{len(dcm_files)}",
                    'all_files': [os.path.relpath(f, patient_mri_dir) for f in dcm_files]
                }
                
                with open(os.path.join(mri_export_dir, 'file_inventory.json'), 'w') as f:
                    json.dump(inventory, f, indent=2)
                
                print(f"    Copied {copied_count}/{len(dcm_files)} DICOM files")
                return copied_count
    
    except Exception as e:
        print(f"    Error copying MRI files: {e}")
    
    return copied_count

def copy_nrrd_segmentation_masks(patient_id, export_patient_dir, nrrd_data):
    """Copy NRRD segmentation masks from both v2 and v3 data structures"""
    if not nrrd_data['found_masks']:
        return 0
    
    seg_export_dir = os.path.join(export_patient_dir, 'Segmentation_Masks_NRRD')
    os.makedirs(seg_export_dir, exist_ok=True)
    
    copied_count = 0
    
    try:
        # Copy v2 format masks
        if 'v2' in nrrd_data['data_versions']:
            # Copy breast masks
            for split in ['train', 'test']:
                for mask_info in nrrd_data['v2_masks']['breast_masks'][split]:
                    try:
                        src_path = mask_info['full_path']
                        dst_filename = f"v2_breast_{split}_{mask_info['filename']}"
                        dst_path = os.path.join(seg_export_dir, dst_filename)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"      Failed to copy v2 breast mask {mask_info['filename']}: {e}")
            
            # Copy dense tissue masks  
            for split in ['train', 'test']:
                for mask_info in nrrd_data['v2_masks']['dense_tissue_masks'][split]:
                    try:
                        src_path = mask_info['full_path']
                        dst_filename = f"v2_dense_tissue_{split}_{mask_info['filename']}"
                        dst_path = os.path.join(seg_export_dir, dst_filename)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        print(f"      Failed to copy v2 dense tissue mask {mask_info['filename']}: {e}")
        
        # Copy v3 format masks
        if 'v3' in nrrd_data['data_versions']:
            # Copy breast masks
            for mask_info in nrrd_data['v3_masks']['breast_masks']:
                try:
                    src_path = mask_info['full_path']
                    dst_filename = f"v3_{mask_info['filename']}"
                    dst_path = os.path.join(seg_export_dir, dst_filename)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f"      Failed to copy v3 breast mask {mask_info['filename']}: {e}")
            
            # Copy dense tissue and vessels masks
            for mask_info in nrrd_data['v3_masks']['dense_and_vessels_masks']:
                try:
                    src_path = mask_info['full_path']
                    dst_filename = f"v3_{mask_info['filename']}"
                    dst_path = os.path.join(seg_export_dir, dst_filename)
                    shutil.copy2(src_path, dst_path)
                    copied_count += 1
                except Exception as e:
                    print(f"      Failed to copy v3 dense+vessels mask {mask_info['filename']}: {e}")
        
        # Save comprehensive segmentation summary
        seg_summary = {
            'total_copied_masks': copied_count,
            'data_versions_found': nrrd_data['data_versions'],
            'v2_data': {
                'breast_train': len(nrrd_data['v2_masks']['breast_masks']['train']),
                'breast_test': len(nrrd_data['v2_masks']['breast_masks']['test']),
                'dense_tissue_train': len(nrrd_data['v2_masks']['dense_tissue_masks']['train']),
                'dense_tissue_test': len(nrrd_data['v2_masks']['dense_tissue_masks']['test']),
                'source_directory': SEGMENTATION_DIR_V2
            },
            'v3_data': {
                'breast_masks': len(nrrd_data['v3_masks']['breast_masks']),
                'dense_and_vessels_masks': len(nrrd_data['v3_masks']['dense_and_vessels_masks']),
                'source_directory': SEGMENTATION_DIR_V3
            },
            'file_naming_conventions': {
                'v2': 'v2_{mask_type}_{split}_{original_filename}',
                'v3': 'v3_{original_filename}'
            },
            'mask_descriptions': {
                'v2_breast': 'Breast tissue masks from v2 dataset (split by train/test)',
                'v2_dense_tissue': 'Dense tissue masks from v2 dataset (split by train/test)',
                'v3_breast': 'Breast tissue masks from v3 dataset (individual patient folders)',
                'v3_dense_and_vessels': 'Combined dense tissue and blood vessel masks from v3 dataset'
            }
        }
        
        with open(os.path.join(seg_export_dir, 'segmentation_summary.json'), 'w') as f:
            json.dump(seg_summary, f, indent=2)
        
        versions_str = ', '.join(nrrd_data['data_versions'])
        print(f"    Copied {copied_count} NRRD segmentation masks (from {versions_str})")
        
    except Exception as e:
        print(f"    Error copying NRRD masks: {e}")
    
    return copied_count

def export_patients(selected_train_patients, selected_test_patients, all_data):
    """Export complete data for selected patients into train and test folders"""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    # Create train and test directories
    train_dir = os.path.join(EXPORT_DIR, 'train')
    test_dir = os.path.join(EXPORT_DIR, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    exported_patients = {'train': {}, 'test': {}}
    total_patients = len(selected_train_patients) + len(selected_test_patients)
    
    print(f"\n{'='*60}")
    print(f"EXPORTING {total_patients} PATIENTS")
    print(f"{'='*60}")
    print(f"Training patients: {len(selected_train_patients)}")
    print(f"Test patients: {len(selected_test_patients)}")
    print()
    
    # Process training patients
    patient_count = 0
    for i, patient_id in enumerate(selected_train_patients):
        patient_count += 1
        print(f"[{patient_count}/{total_patients}] Processing TRAIN Patient {patient_id}")
        print("-" * 40)
        
        try:
            # Extract patient data
            patient_data = extract_patient_data(patient_id, all_data)
            
            # Double-check that patient has segmentation masks
            if not patient_data['nrrd_segmentation_masks']['found_masks']:
                print(f"  ⚠️  Patient {patient_id} has no segmentation masks, skipping...\n")
                continue
            
            # Create patient directory in train folder
            normalized_id = normalize_patient_id(patient_id)
            patient_dir = os.path.join(train_dir, f'Patient_{normalized_id}')
            os.makedirs(patient_dir, exist_ok=True)
            
            # Save patient JSON
            json_path = os.path.join(patient_dir, 'patient_data.json')
            with open(json_path, 'w') as f:
                json.dump(patient_data, f, indent=2, default=str)
            
            # Copy MRI files
            copied_files = copy_mri_files(patient_id, patient_dir)
            patient_data['copied_dicom_files'] = copied_files
            
            # Copy NRRD segmentation masks
            copied_masks = copy_nrrd_segmentation_masks(patient_id, patient_dir, patient_data['nrrd_segmentation_masks'])
            patient_data['copied_nrrd_masks'] = copied_masks
            
            exported_patients['train'][normalized_id] = patient_data
            print(f"  ✓ Successfully exported TRAIN Patient {normalized_id}\n")
            
        except Exception as e:
            print(f"  ✗ Error exporting TRAIN Patient {patient_id}: {e}\n")
    
    # Process test patients
    for i, patient_id in enumerate(selected_test_patients):
        patient_count += 1
        print(f"[{patient_count}/{total_patients}] Processing TEST Patient {patient_id}")
        print("-" * 40)
        
        try:
            # Extract patient data
            patient_data = extract_patient_data(patient_id, all_data)
            
            # Double-check that patient has segmentation masks
            if not patient_data['nrrd_segmentation_masks']['found_masks']:
                print(f"  ⚠️  Patient {patient_id} has no segmentation masks, skipping...\n")
                continue
            
            # Create patient directory in test folder
            normalized_id = normalize_patient_id(patient_id)
            patient_dir = os.path.join(test_dir, f'Patient_{normalized_id}')
            os.makedirs(patient_dir, exist_ok=True)
            
            # Save patient JSON
            json_path = os.path.join(patient_dir, 'patient_data.json')
            with open(json_path, 'w') as f:
                json.dump(patient_data, f, indent=2, default=str)
            
            # Copy MRI files
            copied_files = copy_mri_files(patient_id, patient_dir)
            patient_data['copied_dicom_files'] = copied_files
            
            # Copy NRRD segmentation masks
            copied_masks = copy_nrrd_segmentation_masks(patient_id, patient_dir, patient_data['nrrd_segmentation_masks'])
            patient_data['copied_nrrd_masks'] = copied_masks
            
            exported_patients['test'][normalized_id] = patient_data
            print(f"  ✓ Successfully exported TEST Patient {normalized_id}\n")
            
        except Exception as e:
            print(f"  ✗ Error exporting TEST Patient {patient_id}: {e}\n")
    
    # Save summary files
    try:
        # JSON summary
        with open(os.path.join(EXPORT_DIR, 'all_patients_summary.json'), 'w') as f:
            json.dump(exported_patients, f, indent=2, default=str)
        
        # CSV summary
        summary_data = []
        for split in ['train', 'test']:
            for patient_id, data in exported_patients[split].items():
                nrrd_summary = data['nrrd_segmentation_masks']['summary'] if data['nrrd_segmentation_masks']['found_masks'] else {
                    'total_nrrd_masks': 0, 'mask_types': [], 'data_versions_found': [], 'v2_masks_count': 0, 'v3_masks_count': 0
                }
                
                summary_row = {
                    'Patient_ID': patient_id,
                    'Patient_ID_Full': data['patient_id_full'],
                    'Dataset_Split': data['dataset_split'],
                    'Export_Split': split,
                    'Has_Clinical_Data': bool(data['demographic_clinical']),
                    'Num_Clinical_Fields': len(data['demographic_clinical']),
                    'Has_Imaging_Features': bool(data['imaging_features']),
                    'Num_Imaging_Features': len(data['imaging_features']),
                    'Has_Annotations': bool(data['annotations']),
                    'Num_Annotations': len(data['annotations']),
                    'Total_Annotated_Volume': data['annotation_summary'].get('total_volume', 0),
                    'Has_Density_Assessment': bool(data['density_assessment']),
                    'Num_MRI_File_Mappings': len(data['mri_files']),
                    'Num_Segmentation_Files': len(data['segmentation_files']),
                    'Segmentation_Labels': ', '.join(data['segmentation_labels']),
                    'Copied_DICOM_Files': data.get('copied_dicom_files', 0),
                    'Has_NRRD_Masks': bool(data['nrrd_segmentation_masks']['found_masks']),
                    'Total_NRRD_Masks': nrrd_summary['total_nrrd_masks'],
                    'NRRD_Data_Versions': ', '.join(nrrd_summary['data_versions_found']),
                    'NRRD_v2_Masks': nrrd_summary['v2_masks_count'],
                    'NRRD_v3_Masks': nrrd_summary['v3_masks_count'],
                    'NRRD_Mask_Types': ', '.join(nrrd_summary['mask_types']),
                    'Copied_NRRD_Masks': data.get('copied_nrrd_masks', 0)
                }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(EXPORT_DIR, 'export_summary.csv'), index=False)
        
        print(f"✓ Saved summary files")
        
    except Exception as e:
        print(f"✗ Error saving summary: {e}")
    
    return exported_patients

def main():
    print("Duke Breast Cancer MRI Dataset Exporter")
    print("=" * 50)
    
    # Load all data
    print("\nLoading data files...")
    all_data = load_all_data()
    
    if not all_data:
        print("ERROR: No data loaded!")
        return
    
    # Get available patients
    print("\nFinding available patients...")
    available_patients = get_available_patients()
    
    if not available_patients:
        print("ERROR: No patients found!")
        return
    
    # Filter patients to only those with segmentation masks
    # print("\nFiltering patients with segmentation masks...")
    patients_with_masks = []
    patients_without_masks = []
    
    for patient_id in available_patients:
        # print(f"  Checking patient {patient_id}...", end=" ")
        if check_patient_has_segmentation_masks(patient_id):
            patients_with_masks.append(patient_id)
            # print("✓ Has segmentation masks")
        else:
            patients_without_masks.append(patient_id)
            # print("✗ No segmentation masks")
    
    print(f"\nFound {len(patients_with_masks)} patients with segmentation masks")
    print(f"Found {len(patients_without_masks)} patients without segmentation masks")
    
    if not patients_with_masks:
        print("ERROR: No patients with segmentation masks found!")
        return
    
    # Separate patients by train/test split
    print("\nDetermining train/test splits for patients with masks...")
    train_patients = []
    test_patients = []
    unknown_patients = []
    
    for patient_id in patients_with_masks:
        split = get_dataset_split(patient_id, all_data['train_ids'], all_data['test_ids'])
        if split == 'train':
            train_patients.append(patient_id)
        elif split == 'test':
            test_patients.append(patient_id)
        else:
            unknown_patients.append(patient_id)
    
    print(f"\nDataset split breakdown:")
    print(f"  Train patients with masks: {len(train_patients)}")
    print(f"  Test patients with masks: {len(test_patients)}")
    print(f"  Unknown split patients with masks: {len(unknown_patients)}")
    
    # Select patients from train and test sets
    selected_train = []
    selected_test = []
    
    # Select training patients
    if len(train_patients) < NUM_TRAIN_PATIENTS:
        selected_train = train_patients
        print(f"\nOnly {len(train_patients)} train patients with masks available, selecting all")
    else:
        selected_train = random.sample(train_patients, NUM_TRAIN_PATIENTS)
        print(f"\nRandomly selected {NUM_TRAIN_PATIENTS} training patients from {len(train_patients)} available")
    
    # Select test patients
    if len(test_patients) < NUM_TEST_PATIENTS:
        selected_test = test_patients
        print(f"Only {len(test_patients)} test patients with masks available, selecting all")
    else:
        selected_test = random.sample(test_patients, NUM_TEST_PATIENTS)
        print(f"Randomly selected {NUM_TEST_PATIENTS} test patients from {len(test_patients)} available")
    
    print(f"\nSelected patients breakdown:")
    print(f"  Training patients ({len(selected_train)}): {selected_train}")
    print(f"  Test patients ({len(selected_test)}): {selected_test}")
    print(f"  Total selected: {len(selected_train) + len(selected_test)} patients")
    
    if not selected_train and not selected_test:
        print("ERROR: No patients selected for export!")
        return
    
    if patients_without_masks:
        print(f"\nPatients without segmentation masks (skipped): {patients_without_masks[:10]}")
        if len(patients_without_masks) > 10:
            print(f"... and {len(patients_without_masks) - 10} more")
    
    # Export
    exported = export_patients(selected_train, selected_test, all_data)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"EXPORT COMPLETED")
    print(f"{'='*60}")
    print(f"Location: {EXPORT_DIR}")
    print(f"Training patients exported: {len(exported['train'])} (in train/ folder)")
    print(f"Test patients exported: {len(exported['test'])} (in test/ folder)")
    print(f"Total patients exported: {len(exported['train']) + len(exported['test'])}")
    print(f"\nDataset statistics:")
    print(f"  Total train patients with masks: {len(train_patients)}")
    print(f"  Total test patients with masks: {len(test_patients)}")
    print(f"  Patients without segmentation masks: {len(patients_without_masks)}")
    print(f"\nExport structure:")
    print("  - train/ (training patients)")
    print("  - test/ (test patients)")
    print(f"\nEach patient folder contains:")
    print("  - patient_data.json (comprehensive data)")
    print("  - MRI_DICOM_sample/ (sample DICOM files + inventory)")
    print("  - Segmentation_Masks_NRRD/ (NRRD segmentation masks + summary)")
    print(f"\nSummary files:")
    print("  - all_patients_summary.json")
    print("  - export_summary.csv")

if __name__ == '__main__':
    main()