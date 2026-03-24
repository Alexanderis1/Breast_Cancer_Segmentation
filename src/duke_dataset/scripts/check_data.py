"""
Quick data distribution analysis for DUKE breast cancer dataset
Run this to see what data you have
"""

import json
from pathlib import Path
from collections import Counter

def check_dataset(dataset_dir):
    """Quick check of dataset"""
    
    data_dir = Path(dataset_dir)
    patient_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_dir}")
    print(f"{'='*70}")
    print(f"Total patients: {len(patient_folders)}\n")
    
    grades = []
    subtypes_list = []
    has_bbox = 0
    has_mri = 0
    samples_per_patient = []
    
    for patient_dir in patient_folders:
        data_file = patient_dir / 'patient_data.json'
        
        if not data_file.exists():
            continue
        
        try:
            with open(data_file) as f:
                metadata = json.load(f)
            
            clinical = metadata.get('demographic_clinical', {})
            
            # Grade
            grade = clinical.get('Tumor Grade', 3)
            if grade in [1, 2, 3]:
                grades.append(int(grade))
            
            # Receptor status -> Subtype
            er = int(clinical.get('ER', 0))
            pr = int(clinical.get('PR', 0))
            her2 = int(clinical.get('HER2', 0))
            
            if er == 0 and pr == 0 and her2 == 0:
                subtype = 'Triple Negative'
            elif her2 == 1:
                subtype = 'HER2+'
            else:
                subtype = 'Luminal'
            subtypes_list.append(subtype)
            
            # Check for annotations
            annotations = metadata.get('annotations', [])
            if annotations and 'bounding_box_3d' in annotations[0]:
                has_bbox += 1
            
            # Check MRI
            mri_dir = patient_dir / 'MRI_DICOM_sample'
            if mri_dir.exists():
                dicom_files = list(mri_dir.glob('**/*.dcm'))
                if dicom_files:
                    has_mri += 1
                    samples_per_patient.append(len(dicom_files))
            
        except Exception as e:
            pass
    
    # Print results
    print("📊 CLASS DISTRIBUTION:")
    print("\n  Tumor Grade:")
    for grade in [1, 2, 3]:
        count = sum(1 for g in grades if g == grade)
        pct = (count / len(grades) * 100) if grades else 0
        bar = '█' * (count // 3) if count > 0 else ''
        print(f"    Grade {grade}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print(f"\n  Molecular Subtype:")
    subtype_counts = Counter(subtypes_list)
    for subtype in ['Triple Negative', 'HER2+', 'Luminal']:
        count = subtype_counts.get(subtype, 0)
        pct = (count / len(subtypes_list) * 100) if subtypes_list else 0
        bar = '█' * (count // 3) if count > 0 else ''
        print(f"    {subtype:20s}: {count:3d} ({pct:5.1f}%) {bar}")
    
    print("\n📈 DATA COMPLETENESS:")
    print(f"  Patients with MRI: {has_mri}/{len(patient_folders)} ({has_mri/len(patient_folders)*100:.1f}%)")
    print(f"  Patients with BBox: {has_bbox}/{len(patient_folders)} ({has_bbox/len(patient_folders)*100:.1f}%)")
    
    if samples_per_patient:
        avg_slices = sum(samples_per_patient) / len(samples_per_patient)
        min_slices = min(samples_per_patient)
        max_slices = max(samples_per_patient)
        print(f"  MRI Slices per patient:")
        print(f"    Average: {avg_slices:.1f}")
        print(f"    Min: {min_slices}, Max: {max_slices}")
    
    return len(patient_folders), has_bbox, has_mri


if __name__ == '__main__':
    print("\n" + "="*70)
    print("DUKE DATASET QUICK CHECK")
    print("="*70)
    
    train_count, train_bbox, train_mri = check_dataset('exported_patients/train')
    test_count, test_bbox, test_mri = check_dataset('exported_patients/test')
    
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Training set: {train_count} patients ({train_mri} with MRI, {train_bbox} with BBox)")
    print(f"Test set:     {test_count} patients ({test_mri} with MRI, {test_bbox} with BBox)")
    print(f"Total:        {train_count + test_count} patients")
    print(f"{'='*70}\n")
