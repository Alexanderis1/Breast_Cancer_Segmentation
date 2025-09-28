import pandas as pd
import os
import random
import json
import shutil
from pathlib import Path

# Dataset paths
DATA_DIR = ('/mnt/c/Users/35193/Desktop/duke_dataset/data')
OUT_DIR = os.getcwd()
EXPORT_DIR = os.path.join(OUT_DIR, 'exported_10_patients_v4')  # New version
NUM_PATIENTS = 10

def debug_dataframe_structure(df, name):
    """Debug helper to show DataFrame structure"""
    if df.empty:
        print(f"    {name} is empty")
        return
    
    print(f"    {name} shape: {df.shape}")
    print(f"    {name} columns: {df.columns.tolist()}")
    
    # Show sample patient IDs for key columns
    patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
    for col in patient_id_cols:
        if col in df.columns:
            unique_vals = df[col].dropna().unique()
            print(f"    {name}['{col}'] sample values: {unique_vals[:10]}")
            print(f"    {name}['{col}'] total unique: {len(unique_vals)}")
            break

def load_all_data():
    """Load all Excel and CSV files containing patient data"""
    data = {}
    
    print(f"Looking for data files in: {DATA_DIR}")
    
    # Check if DATA_DIR exists
    if not os.path.exists(DATA_DIR):
        print(f"❌ DATA_DIR does not exist: {DATA_DIR}")
        print("Available directories:")
        parent_dir = os.path.dirname(DATA_DIR)
        if os.path.exists(parent_dir):
            for item in os.listdir(parent_dir):
                print(f"  {item}")
        return {}
    
    # Load Excel files with error handling
    excel_files = {
        'annotation_boxes': 'Annotation_Boxes.xlsx',
        'density_assessments': 'Breast_Radiologist_Density_Assessments.xlsx',
        'filepath_mapping': 'Breast-Cancer-MRI-filepath_filename-mapping.xlsx',
        'clinical_features': 'Clinical_and_Other_Features.xlsx',
        'imaging_features': 'Imaging_Features.xlsx'
    }
    
    for key, filename in excel_files.items():
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(file_path):
            try:
                # Read Excel file
                df = pd.read_excel(file_path)
                data[key] = df
                print(f"✓ Loaded {key}: {len(df)} rows")
                debug_dataframe_structure(df, key)
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"✗ File not found: {filename}")
            data[key] = pd.DataFrame()
    
    # Load CSV files
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
                debug_dataframe_structure(df, key)
                
                # Show sample for segmentation mapping
                if key == 'segmentation_mapping':
                    print(f"    Sample segmentation labels: {df['Segmentation Label'].unique()[:5]}")
                    
            except Exception as e:
                print(f"✗ Error loading {filename}: {e}")
                data[key] = pd.DataFrame()
        else:
            print(f"✗ File not found: {filename}")
            data[key] = pd.DataFrame()
    
    return data

def get_available_patients():
    """Get list of all available patient IDs from MRI folders and data files"""
    patients = set()
    
    try:
        # Get patients from MRI directories
        manifest_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith('manifest-')]
        print(f"Found manifest directories: {manifest_dirs}")
        
        for manifest_dir in manifest_dirs:
            duke_mri_path = os.path.join(DATA_DIR, manifest_dir, 'Duke-Breast-Cancer-MRI')
            if os.path.exists(duke_mri_path):
                for folder in os.listdir(duke_mri_path):
                    if folder.startswith('Breast_MRI_') and os.path.isdir(os.path.join(duke_mri_path, folder)):
                        patient_id = folder.replace('Breast_MRI_', '')
                        patients.add(patient_id)
        
        print(f"Found {len(patients)} patients from MRI directories")
        
        # Also get patients from data files
        data_files_to_check = [
            'segmentation_filepath_mapping.csv',
            'Clinical_and_Other_Features.xlsx',
            'Imaging_Features.xlsx'
        ]
        
        for filename in data_files_to_check:
            file_path = os.path.join(DATA_DIR, filename)
            if os.path.exists(file_path):
                try:
                    if filename.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    # Look for patient ID columns
                    patient_id_cols = ['Patient ID', 'MRN', 'Patient_ID', 'ID', 'PatientID']
                    
                    for col in patient_id_cols:
                        if col in df.columns:
                            if col == 'Patient ID':  # From segmentation file
                                # Extract patient numbers from "Breast_MRI_XXX" format
                                patient_ids = df[col].str.replace('Breast_MRI_', '').unique()
                                for pid in patient_ids:
                                    if str(pid).isdigit():
                                        patients.add(str(pid))
                            else:
                                # Direct patient IDs
                                for pid in df[col].dropna().unique():
                                    if str(pid).isdigit():
                                        patients.add(str(pid))
                            break
                            
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
    except Exception as e:
        print(f"Error finding patients: {e}")
    
    return sorted(list(patients))

def extract_patient_data(patient_id, all_data):
    """Extract all data for a specific patient from all data sources"""
    patient_data = {
        'patient_id': patient_id,
        'demographic_clinical': {},
        'imaging_features': {},
        'annotations': [],
        'annotation_summary': {},
        'density_assessment': {},
        'mri_files': [],
        'segmentation_files': [],
        'segmentation_labels': [],
        'dataset_split': 'unknown'
    }
    
    try:
        print(f"  Extracting data for patient {patient_id}...")
        
        # Extract clinical features - try multiple matching strategies
        if not all_data['clinical_features'].empty:
            df = all_data['clinical_features']
            print(f"    Clinical features DataFrame shape: {df.shape}")
            print(f"    Clinical features columns: {df.columns.tolist()}")
            
            # Try different patient ID formats and column names
            patient_id_variations = [
                str(patient_id),
                f"Patient_{patient_id}",
                f"Breast_MRI_{patient_id}",
                int(patient_id) if patient_id.isdigit() else patient_id
            ]
            
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
            
            found_clinical = False
            for col in patient_id_cols:
                if col in df.columns:
                    print(f"    Checking column '{col}' with unique values: {df[col].unique()[:10]}")
                    for pid_var in patient_id_variations:
                        try:
                            if col == 'MRN' or 'mrn' in col.lower():
                                # For MRN columns, try exact match
                                mask = df[col] == pid_var
                            else:
                                # For other columns, try string comparison
                                mask = df[col].astype(str).str.contains(str(pid_var), na=False, case=False)
                            
                            if mask.any():
                                clinical_data = df[mask].iloc[0].to_dict()
                                # Clean up the data (remove NaN values)
                                patient_data['demographic_clinical'] = {k: v for k, v in clinical_data.items() 
                                                                     if pd.notna(v) and str(v) != 'nan'}
                                print(f"    ✓ Found clinical data using {col}={pid_var}")
                                print(f"    ✓ Clinical data keys: {list(patient_data['demographic_clinical'].keys())}")
                                found_clinical = True
                                break
                        except Exception as e:
                            continue
                    if found_clinical:
                        break
            
            if not found_clinical:
                print(f"    ✗ No clinical data found for patient {patient_id}")
        
        # Extract imaging features - improved matching
        if not all_data['imaging_features'].empty:
            df = all_data['imaging_features']
            print(f"    Imaging features DataFrame shape: {df.shape}")
            print(f"    Imaging features columns: {df.columns.tolist()}")
            
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
            
            found_imaging = False
            for col in patient_id_cols:
                if col in df.columns:
                    print(f"    Checking imaging column '{col}' with unique values: {df[col].unique()[:10]}")
                    for pid_var in patient_id_variations:
                        try:
                            if col == 'MRN' or 'mrn' in col.lower():
                                mask = df[col] == pid_var
                            else:
                                mask = df[col].astype(str).str.contains(str(pid_var), na=False, case=False)
                            
                            if mask.any():
                                imaging_data = df[mask].iloc[0].to_dict()
                                # Clean up the data
                                patient_data['imaging_features'] = {k: v for k, v in imaging_data.items() 
                                                                  if pd.notna(v) and str(v) != 'nan'}
                                print(f"    ✓ Found imaging features using {col}={pid_var}")
                                print(f"    ✓ Found {len(patient_data['imaging_features'])} imaging features")
                                found_imaging = True
                                break
                        except Exception as e:
                            continue
                    if found_imaging:
                        break
            
            if not found_imaging:
                print(f"    ✗ No imaging features found for patient {patient_id}")
        
        # Extract segmentation data
        if not all_data['segmentation_mapping'].empty:
            df = all_data['segmentation_mapping']
            if 'Patient ID' in df.columns:
                mask = df['Patient ID'] == f'Breast_MRI_{patient_id}'
                if mask.any():
                    seg_data = df[mask]
                    patient_data['segmentation_files'] = seg_data.to_dict('records')
                    patient_data['segmentation_labels'] = seg_data['Segmentation Label'].unique().tolist()
                    print(f"    ✓ Found {len(seg_data)} segmentation entries with labels: {patient_data['segmentation_labels']}")
        
        # Extract annotations (bounding boxes) - comprehensive matching
        if not all_data['annotation_boxes'].empty:
            df = all_data['annotation_boxes']
            print(f"    Annotations DataFrame shape: {df.shape}")
            print(f"    Annotations columns: {df.columns.tolist()}")
            
            # Show sample of annotation data structure
            if len(df) > 0:
                print(f"    Sample annotation data: {df.head(1).to_dict('records')}")
            
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1', 'patient_mrn']
            
            found_annotations = False
            for col in patient_id_cols:
                if col in df.columns:
                    print(f"    Checking annotations column '{col}' with unique values: {df[col].unique()[:10]}")
                    for pid_var in patient_id_variations:
                        try:
                            if col == 'MRN' or 'mrn' in col.lower():
                                # For MRN columns, try exact match with different formats
                                mask = (df[col] == pid_var) | (df[col].astype(str) == str(pid_var))
                            else:
                                # For other columns, try multiple matching strategies
                                mask = (
                                    (df[col].astype(str) == str(pid_var)) |
                                    (df[col].astype(str).str.contains(str(pid_var), na=False, case=False)) |
                                    (df[col].astype(str).str.endswith(str(pid_var), na=False))
                                )
                            
                            if mask.any():
                                matched_rows = df[mask]
                                print(f"    ✓ Found {len(matched_rows)} annotation rows using {col}={pid_var}")
                                
                                # Process annotations with detailed structure
                                annotations = []
                                for _, row in matched_rows.iterrows():
                                    ann_dict = {}
                                    for k, v in row.items():
                                        if pd.notna(v) and str(v) != 'nan' and str(v).strip() != '':
                                            ann_dict[k] = v
                                    
                                    # Ensure we capture bounding box coordinates if they exist
                                    bbox_fields = ['x', 'y', 'width', 'height', 'x1', 'y1', 'x2', 'y2', 
                                                 'left', 'top', 'right', 'bottom', 'bbox_x', 'bbox_y', 
                                                 'bbox_width', 'bbox_height']
                                    bbox_data = {}
                                    for field in bbox_fields:
                                        for col_name in ann_dict.keys():
                                            if field.lower() in col_name.lower():
                                                bbox_data[field] = ann_dict[col_name]
                                    
                                    if bbox_data:
                                        ann_dict['bounding_box'] = bbox_data
                                    
                                    if ann_dict:  # Only add if not empty
                                        annotations.append(ann_dict)
                                
                                patient_data['annotations'] = annotations
                                
                                # Extract annotation summary
                                if annotations:
                                    annotation_types = set()
                                    for ann in annotations:
                                        for key in ann.keys():
                                            if 'type' in key.lower() or 'label' in key.lower() or 'class' in key.lower():
                                                if ann[key]:
                                                    annotation_types.add(str(ann[key]))
                                    
                                    patient_data['annotation_summary'] = {
                                        'total_annotations': len(annotations),
                                        'annotation_types': list(annotation_types),
                                        'has_bounding_boxes': any('bounding_box' in ann for ann in annotations)
                                    }
                                    
                                    print(f"    ✓ Processed {len(annotations)} annotations")
                                    print(f"    ✓ Annotation types found: {list(annotation_types)}")
                                    print(f"    ✓ Has bounding boxes: {patient_data['annotation_summary']['has_bounding_boxes']}")
                                
                                found_annotations = True
                                break
                        except Exception as e:
                            print(f"    Error processing annotations for {col}={pid_var}: {e}")
                            continue
                    if found_annotations:
                        break
            
            if not found_annotations:
                print(f"    ✗ No annotations found for patient {patient_id}")
                # Initialize empty annotation summary
                patient_data['annotation_summary'] = {
                    'total_annotations': 0,
                    'annotation_types': [],
                    'has_bounding_boxes': False
                }
        
        # Extract density assessment - improved matching
        if not all_data['density_assessments'].empty:
            df = all_data['density_assessments']
            print(f"    Density assessments DataFrame shape: {df.shape}")
            print(f"    Density assessments columns: {df.columns.tolist()}")
            
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
            
            found_density = False
            for col in patient_id_cols:
                if col in df.columns:
                    print(f"    Checking density column '{col}' with unique values: {df[col].unique()[:10]}")
                    for pid_var in patient_id_variations:
                        try:
                            if col == 'MRN' or 'mrn' in col.lower():
                                mask = df[col] == pid_var
                            else:
                                mask = df[col].astype(str).str.contains(str(pid_var), na=False, case=False)
                            
                            if mask.any():
                                density_data = df[mask].iloc[0].to_dict()
                                patient_data['density_assessment'] = {k: v for k, v in density_data.items() 
                                                                    if pd.notna(v) and str(v) != 'nan'}
                                print(f"    ✓ Found density assessment using {col}={pid_var}")
                                found_density = True
                                break
                        except Exception as e:
                            continue
                    if found_density:
                        break
            
            if not found_density:
                print(f"    ✗ No density assessment found for patient {patient_id}")
        
        # Extract MRI file mappings - improved matching
        if not all_data['filepath_mapping'].empty:
            df = all_data['filepath_mapping']
            print(f"    MRI filepath mapping DataFrame shape: {df.shape}")
            print(f"    MRI filepath mapping columns: {df.columns.tolist()}")
            
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
            
            found_mri = False
            for col in patient_id_cols:
                if col in df.columns:
                    print(f"    Checking MRI column '{col}' with unique values: {df[col].unique()[:10]}")
                    for pid_var in patient_id_variations:
                        try:
                            if col == 'MRN' or 'mrn' in col.lower():
                                mask = df[col] == pid_var
                            else:
                                mask = df[col].astype(str).str.contains(str(pid_var), na=False, case=False)
                            
                            if mask.any():
                                mri_files = df[mask].to_dict('records')
                                patient_data['mri_files'] = [
                                    {k: v for k, v in mri.items() if pd.notna(v) and str(v) != 'nan'}
                                    for mri in mri_files
                                ]
                                print(f"    ✓ Found {len(mri_files)} MRI file mappings using {col}={pid_var}")
                                found_mri = True
                                break
                        except Exception as e:
                            continue
                    if found_mri:
                        break
            
            if not found_mri:
                print(f"    ✗ No MRI file mappings found for patient {patient_id}")
        
        # Check train/test split - improved matching
        for split_name in ['train_ids', 'test_ids']:
            if not all_data[split_name].empty:
                df = all_data[split_name]
                print(f"    Checking {split_name} DataFrame shape: {df.shape}")
                print(f"    {split_name} columns: {df.columns.tolist()}")
                
                patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id', 'Patient ID', 'MRN.1']
                
                for col in patient_id_cols:
                    if col in df.columns:
                        print(f"    Checking {split_name} column '{col}' with unique values: {df[col].unique()[:10]}")
                        for pid_var in patient_id_variations:
                            try:
                                if col == 'MRN' or 'mrn' in col.lower():
                                    if pid_var in df[col].values:
                                        patient_data['dataset_split'] = split_name.replace('_ids', '')
                                        print(f"    ✓ Found in {split_name} using {col}={pid_var}")
                                        break
                                else:
                                    mask = df[col].astype(str).str.contains(str(pid_var), na=False, case=False)
                                    if mask.any():
                                        patient_data['dataset_split'] = split_name.replace('_ids', '')
                                        print(f"    ✓ Found in {split_name} using {col}={pid_var}")
                                        break
                            except Exception as e:
                                continue
                        if patient_data['dataset_split'] != 'unknown':
                            break
                
                if patient_data['dataset_split'] != 'unknown':
                    break
        
        if patient_data['dataset_split'] == 'unknown':
            print(f"    ✗ Dataset split not found for patient {patient_id}")
    
    except Exception as e:
        print(f"    ✗ Error extracting data for patient {patient_id}: {e}")
    
    return patient_data

def copy_sample_mri_files(patient_id, export_patient_dir, max_files=20):
    """Copy a sample of MRI DICOM files for the patient"""
    mri_export_dir = os.path.join(export_patient_dir, 'MRI_DICOM_sample')
    os.makedirs(mri_export_dir, exist_ok=True)
    
    copied_count = 0
    
    try:
        # Look for patient MRI folder in manifest directories
        manifest_dirs = [d for d in os.listdir(DATA_DIR) if d.startswith('manifest-')]
        
        for manifest_dir in manifest_dirs:
            patient_mri_dir = os.path.join(DATA_DIR, manifest_dir, 'Duke-Breast-Cancer-MRI', f'Breast_MRI_{patient_id}')
            
            if os.path.exists(patient_mri_dir):
                print(f"    Found MRI directory: {os.path.basename(patient_mri_dir)}")
                
                # Get all DICOM files
                dcm_files = []
                for root, dirs, files in os.walk(patient_mri_dir):
                    for file in files:
                        if file.lower().endswith('.dcm'):
                            dcm_files.append(os.path.join(root, file))
                
                # Copy a sample
                sample_files = dcm_files[:max_files] if len(dcm_files) > max_files else dcm_files
                
                for src_path in sample_files:
                    try:
                        rel_path = os.path.relpath(src_path, patient_mri_dir)
                        dst_path = os.path.join(mri_export_dir, rel_path)
                        
                        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                        
                    except Exception as e:
                        print(f"      Failed to copy {os.path.basename(src_path)}: {e}")
                
                # Create file inventory
                file_inventory = {
                    'total_dcm_files': len(dcm_files),
                    'copied_files': copied_count,
                    'sample_ratio': f"{copied_count}/{len(dcm_files)}",
                    'all_files_list': [os.path.relpath(f, patient_mri_dir) for f in dcm_files]
                }
                
                with open(os.path.join(mri_export_dir, 'file_inventory.json'), 'w') as f:
                    json.dump(file_inventory, f, indent=2)
                
                print(f"    ✓ Copied {copied_count}/{len(dcm_files)} DICOM files")
                return copied_count
        
        print(f"    ✗ No MRI directory found for patient {patient_id}")
        
    except Exception as e:
        print(f"    ✗ Error copying MRI files: {e}")
    
    return copied_count

def export_selected_patients(selected_patient_ids, all_data):
    """Export complete data for selected patients"""
    os.makedirs(EXPORT_DIR, exist_ok=True)
    exported_patients = {}
    
    print(f"\n{'='*60}")
    print(f"EXPORTING {len(selected_patient_ids)} PATIENTS")
    print(f"{'='*60}")
    
    for i, patient_id in enumerate(selected_patient_ids):
        print(f"\n[{i+1}/{len(selected_patient_ids)}] Processing Patient {patient_id}")
        print("-" * 40)
        
        try:
            # Extract all patient data
            patient_data = extract_patient_data(patient_id, all_data)
            exported_patients[patient_id] = patient_data
            
            # Create patient directory
            patient_export_dir = os.path.join(EXPORT_DIR, f'Patient_{patient_id}')
            os.makedirs(patient_export_dir, exist_ok=True)
            
            # Save patient JSON
            with open(os.path.join(patient_export_dir, 'patient_data.json'), 'w') as f:
                json.dump(patient_data, f, indent=2, default=str)
            
            # Copy sample MRI files
            copied_files = copy_sample_mri_files(patient_id, patient_export_dir)
            patient_data['copied_dicom_files'] = copied_files
            
            # Show summary of extracted data
            print(f"  📊 EXTRACTION SUMMARY for Patient {patient_id}:")
            print(f"      Clinical data: {'✓' if patient_data['demographic_clinical'] else '✗'} ({len(patient_data['demographic_clinical'])} fields)")
            print(f"      Imaging features: {'✓' if patient_data['imaging_features'] else '✗'} ({len(patient_data['imaging_features'])} features)")
            print(f"      Annotations: {'✓' if patient_data['annotations'] else '✗'} ({len(patient_data['annotations'])} entries)")
            if patient_data['annotation_summary'].get('total_annotations', 0) > 0:
                print(f"        - Annotation types: {patient_data['annotation_summary']['annotation_types']}")
                print(f"        - Has bounding boxes: {patient_data['annotation_summary']['has_bounding_boxes']}")
            print(f"      Density assessment: {'✓' if patient_data['density_assessment'] else '✗'} ({len(patient_data['density_assessment'])} fields)")
            print(f"      MRI files: {'✓' if patient_data['mri_files'] else '✗'} ({len(patient_data['mri_files'])} files)")
            print(f"      Segmentation files: {'✓' if patient_data['segmentation_files'] else '✗'} ({len(patient_data['segmentation_files'])} files)")
            print(f"      Dataset split: {patient_data['dataset_split']}")
            print(f"      DICOM files copied: {copied_files}")
            
            print(f"  ✅ Successfully exported Patient {patient_id}")
            
        except Exception as e:
            print(f"  ❌ Error exporting Patient {patient_id}: {e}")
    
    # Save summary
    try:
        with open(os.path.join(EXPORT_DIR, 'all_patients_summary.json'), 'w') as f:
            json.dump(exported_patients, f, indent=2, default=str)
        
        # Create CSV summary
        summary_data = []
        for patient_id, data in exported_patients.items():
            summary_row = {
                'Patient_ID': patient_id,
                'Dataset_Split': data.get('dataset_split', 'unknown'),
                'Has_Clinical_Data': bool(data.get('demographic_clinical')),
                'Has_Imaging_Features': bool(data.get('imaging_features')),
                'Num_Imaging_Features': len(data.get('imaging_features', {})),
                'Has_Annotations': bool(data.get('annotations')),
                'Num_Annotations': len(data.get('annotations', [])),
                'Annotation_Types': ', '.join(data.get('annotation_summary', {}).get('annotation_types', [])),
                'Has_Bounding_Boxes': data.get('annotation_summary', {}).get('has_bounding_boxes', False),
                'Has_Density_Assessment': bool(data.get('density_assessment')),
                'Num_Segmentation_Files': len(data.get('segmentation_files', [])),
                'Segmentation_Labels': ', '.join(data.get('segmentation_labels', [])),
                'Copied_DICOM_Files': data.get('copied_dicom_files', 0)
            }
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(EXPORT_DIR, 'export_summary.csv'), index=False)
        
        print(f"\n✅ Saved summary files")
        
    except Exception as e:
        print(f"❌ Error saving summary: {e}")
    
    return exported_patients

def main():
    print("🔬 Duke Breast Cancer MRI Dataset Analyzer")
    print("=" * 50)
    
    # Load all data
    print("\n📂 Loading data files...")
    all_data = load_all_data()
    
    # Get available patients
    print("\n🔍 Finding available patients...")
    available_patients = get_available_patients()
    print(f"Found {len(available_patients)} total patients")
    
    if len(available_patients) == 0:
        print("❌ No patients found! Check data structure.")
        return
    
    # Select patients
    if len(available_patients) < NUM_PATIENTS:
        selected_patients = available_patients
        print(f"⚠️  Only {len(available_patients)} patients available, selecting all")
    else:
        selected_patients = random.sample(available_patients, NUM_PATIENTS)
        print(f"🎯 Randomly selected {NUM_PATIENTS} patients")
    
    print(f"Selected patients: {selected_patients}")
    
    # Export patients
    exported_data = export_selected_patients(selected_patients, all_data)
    
    # Final summary
    print(f"\n🎉 EXPORT COMPLETED!")
    print(f"📁 Location: {EXPORT_DIR}")
    print(f"👥 Patients exported: {len(exported_data)}")
    print(f"\n📋 Each patient folder contains:")
    print("   • patient_data.json (comprehensive data)")
    print("   • MRI_DICOM_sample/ (sample DICOM files + inventory)")
    print(f"\n📊 Summary files:")
    print("   • all_patients_summary.json")
    print("   • export_summary.csv")

if __name__ == '__main__':
    main()