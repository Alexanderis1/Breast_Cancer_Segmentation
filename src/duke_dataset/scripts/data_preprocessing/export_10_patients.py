import pandas as pd
import os
import random
import json
import shutil
from pathlib import Path

# Dataset paths
DATA_DIR = os.getcwd()
EXPORT_DIR = os.path.join(DATA_DIR, 'exported_10_patients_v3')
NUM_PATIENTS = 10

def load_all_data():
    """Load all Excel and CSV files containing patient data"""
    data = {}
    
    print(f"Looking for data files in: {DATA_DIR}")
    
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
                print(f"✓ Loaded {key}: {len(df)} rows, columns: {list(df.columns)}")
                
                # Show sample data for first file
                if key == 'clinical_features':
                    print(f"  Sample columns: {df.columns.tolist()[:10]}")
                    
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
                print(f"✓ Loaded {key}: {len(df)} rows, columns: {list(df.columns)}")
                
                # Show sample for segmentation mapping
                if key == 'segmentation_mapping':
                    print(f"  Sample segmentation labels: {df['Segmentation Label'].unique()[:5]}")
                    
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
        'annotations': {},
        'density_assessment': {},
        'mri_files': [],
        'segmentation_files': [],
        'segmentation_labels': [],
        'dataset_split': 'unknown'
    }
    
    try:
        print(f"  Extracting data for patient {patient_id}...")
        
        # Extract clinical features
        if not all_data['clinical_features'].empty:
            df = all_data['clinical_features']
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
            
            for col in patient_id_cols:
                if col in df.columns:
                    mask = df[col].astype(str) == str(patient_id)
                    if mask.any():
                        patient_data['demographic_clinical'] = df[mask].iloc[0].to_dict()
                        print(f"    ✓ Found clinical data")
                        break
        
        # Extract imaging features  
        if not all_data['imaging_features'].empty:
            df = all_data['imaging_features']
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
            
            for col in patient_id_cols:
                if col in df.columns:
                    mask = df[col].astype(str) == str(patient_id)
                    if mask.any():
                        patient_data['imaging_features'] = df[mask].iloc[0].to_dict()
                        print(f"    ✓ Found imaging features ({len(df[mask].iloc[0].to_dict())} features)")
                        break
        
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
        
        # Extract annotations
        if not all_data['annotation_boxes'].empty:
            df = all_data['annotation_boxes']
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
            
            for col in patient_id_cols:
                if col in df.columns:
                    mask = df[col].astype(str) == str(patient_id)
                    if mask.any():
                        patient_data['annotations'] = df[mask].to_dict('records')
                        print(f"    ✓ Found {len(df[mask])} annotation entries")
                        break
        
        # Extract density assessment
        if not all_data['density_assessments'].empty:
            df = all_data['density_assessments']
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
            
            for col in patient_id_cols:
                if col in df.columns:
                    mask = df[col].astype(str) == str(patient_id)
                    if mask.any():
                        patient_data['density_assessment'] = df[mask].iloc[0].to_dict()
                        print(f"    ✓ Found density assessment")
                        break
        
        # Extract MRI file mappings
        if not all_data['filepath_mapping'].empty:
            df = all_data['filepath_mapping']
            patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
            
            for col in patient_id_cols:
                if col in df.columns:
                    mask = df[col].astype(str) == str(patient_id)
                    if mask.any():
                        patient_data['mri_files'] = df[mask].to_dict('records')
                        print(f"    ✓ Found {len(df[mask])} MRI file mappings")
                        break
        
        # Check train/test split
        for split_name in ['train_ids', 'test_ids']:
            if not all_data[split_name].empty:
                df = all_data[split_name]
                patient_id_cols = ['MRN', 'Patient_ID', 'ID', 'PatientID', 'patient_id']
                
                for col in patient_id_cols:
                    if col in df.columns:
                        if int(patient_id) in df[col].values:
                            patient_data['dataset_split'] = split_name.replace('_ids', '')
                            print(f"    ✓ Found in {split_name}")
                            break
                
                if patient_data['dataset_split'] != 'unknown':
                    break
    
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