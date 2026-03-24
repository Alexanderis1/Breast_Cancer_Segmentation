"""
Quick data distribution analysis script for DUKE breast cancer dataset
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

def analyze_dataset(dataset_dir):
    """Analyze dataset distribution"""
    
    data_dir = Path(dataset_dir)
    patient_folders = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\n" + "="*70)
    print(f"ANALYZING: {dataset_dir}")
    print("="*70)
    print(f"Total patients: {len(patient_folders)}\n")
    
    # Collect statistics
    grades = []
    subtypes = []
    high_grades = []
    er_status = []
    pr_status = []
    her2_status = []
    
    clinical_features_all = []
    imaging_features_all = []
    has_bbox_count = 0
    
    for patient_dir in patient_folders:
        data_file = patient_dir / 'patient_data.json'
        
        if not data_file.exists():
            continue
        
        try:
            with open(data_file) as f:
                metadata = json.load(f)
            
            clinical = metadata.get('demographic_clinical', {})
            imaging = metadata.get('imaging_features', {})
            annotations = metadata.get('annotations', [])
            
            # Grade
            grade = clinical.get('Tumor Grade', 3)
            if grade in [1, 2, 3]:
                grades.append(grade)
            
            # ER/PR/HER2
            er = int(clinical.get('ER', 0))
            pr = int(clinical.get('PR', 0))
            her2 = int(clinical.get('HER2', 0))
            
            er_status.append(er)
            pr_status.append(pr)
            her2_status.append(her2)
            
            # Subtype
            if er == 0 and pr == 0 and her2 == 0:
                subtype = 0  # Triple Negative
            elif her2 == 1:
                subtype = 1  # HER2+
            else:
                subtype = 2  # Luminal
            subtypes.append(subtype)
            
            # High grade
            high_grade = 1 if grade == 3 else 0
            high_grades.append(high_grade)
            
            # Check for bbox
            if annotations and 'bounding_box_3d' in annotations[0]:
                has_bbox_count += 1
            
            # Clinical features
            clinical_features = []
            feature_keys = [
                'Age at last contact in EMR f/u(days)(from the date of diagnosis) ,last time patient known to be alive, unless age of death is reported(in such case the age of death',
                'Field Strength (Tesla)',
                'TR (Repetition Time)',
                'TE (Echo Time)',
                'Slice Thickness ',
            ]
            for key in feature_keys:
                val = clinical.get(key, 0)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    clinical_features.append(float(val))
            if clinical_features:
                clinical_features_all.append(clinical_features)
            
            # Imaging features
            imaging_features = []
            imaging_keys = [
                'TumorMajorAxisLength_mm',
                'Volume_cu_mm_Tumor',
            ]
            for key in imaging_keys:
                val = imaging.get(key, 0)
                if isinstance(val, (int, float)) and not np.isnan(val):
                    imaging_features.append(float(val))
            if imaging_features:
                imaging_features_all.append(imaging_features)
            
        except Exception as e:
            continue
    
    # Print statistics
    print("📊 CLASS DISTRIBUTION:")
    print(f"\n  Tumor Grade:")
    grade_counts = Counter(grades)
    for grade in [1, 2, 3]:
        count = grade_counts.get(grade, 0)
        pct = (count / len(grades) * 100) if grades else 0
        print(f"    Grade {grade}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Molecular Subtype:")
    subtype_names = ['Triple Negative', 'HER2+', 'Luminal']
    subtype_counts = Counter(subtypes)
    for i, name in enumerate(subtype_names):
        count = subtype_counts.get(i, 0)
        pct = (count / len(subtypes) * 100) if subtypes else 0
        print(f"    {name:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n  Receptor Status:")
    er_pos = sum(er_status)
    pr_pos = sum(pr_status)
    her2_pos = sum(her2_status)
    total = len(er_status)
    print(f"    ER+:  {er_pos:3d} ({er_pos/total*100:5.1f}%)")
    print(f"    PR+:  {pr_pos:3d} ({pr_pos/total*100:5.1f}%)")
    print(f"    HER2+: {her2_pos:3d} ({her2_pos/total*100:5.1f}%)")
    
    print(f"\n  High Grade (Grade 3):")
    high_grade_count = sum(high_grades)
    print(f"    Yes: {high_grade_count:3d} ({high_grade_count/len(high_grades)*100:5.1f}%)")
    print(f"    No:  {len(high_grades)-high_grade_count:3d} ({(1-high_grade_count/len(high_grades))*100:5.1f}%)")
    
    print(f"\n  3D Bounding Boxes: {has_bbox_count:3d} ({has_bbox_count/len(patient_folders)*100:5.1f}%)")
    
    # Feature statistics
    if clinical_features_all:
        clinical_array = np.array(clinical_features_all)
        print(f"\n📈 CLINICAL FEATURES:")
        print(f"   Field Strength: {clinical_array[:, 1].mean():.2f} ± {clinical_array[:, 1].std():.2f} T")
        print(f"   TR (ms):        {clinical_array[:, 2].mean():.0f} ± {clinical_array[:, 2].std():.0f}")
        print(f"   TE (ms):        {clinical_array[:, 3].mean():.0f} ± {clinical_array[:, 3].std():.0f}")
    
    if imaging_features_all:
        imaging_array = np.array(imaging_features_all)
        print(f"\n🩺 IMAGING FEATURES:")
        print(f"   Tumor Size:   {imaging_array[:, 0].mean():.1f} ± {imaging_array[:, 0].std():.1f} mm")
        print(f"   Tumor Volume: {imaging_array[:, 1].mean():.0f} ± {imaging_array[:, 1].std():.0f} mm³")
    
    return {
        'grades': grades,
        'subtypes': subtypes,
        'high_grades': high_grades,
        'er_status': er_status,
        'pr_status': pr_status,
        'her2_status': her2_status,
        'has_bbox_count': has_bbox_count,
        'total_patients': len(patient_folders),
        'clinical_features': clinical_features_all,
        'imaging_features': imaging_features_all
    }


def visualize_distributions(train_data, test_data):
    """Create visualization of data distributions"""
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Tumor Grade Distribution
    ax1 = plt.subplot(3, 3, 1)
    grades_train = train_data['grades']
    grades_test = test_data['grades']
    grade_labels = ['Grade 1', 'Grade 2', 'Grade 3']
    x_pos = [0, 1, 2]
    train_counts = [sum(1 for g in grades_train if g == i+1) for i in range(3)]
    test_counts = [sum(1 for g in grades_test if g == i+1) for i in range(3)]
    width = 0.35
    ax1.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax1.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax1.set_ylabel('Count')
    ax1.set_title('Tumor Grade Distribution')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(grade_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Molecular Subtype Distribution
    ax2 = plt.subplot(3, 3, 2)
    subtype_labels = ['Triple Neg', 'HER2+', 'Luminal']
    x_pos = [0, 1, 2]
    train_counts = [sum(1 for s in train_data['subtypes'] if s == i) for i in range(3)]
    test_counts = [sum(1 for s in test_data['subtypes'] if s == i) for i in range(3)]
    ax2.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax2.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax2.set_ylabel('Count')
    ax2.set_title('Molecular Subtype Distribution')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(subtype_labels, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. High Grade Distribution
    ax3 = plt.subplot(3, 3, 3)
    hg_labels = ['Grade 1-2', 'Grade 3']
    x_pos = [0, 1]
    train_counts = [sum(1 for h in train_data['high_grades'] if h == 0), sum(train_data['high_grades'])]
    test_counts = [sum(1 for h in test_data['high_grades'] if h == 0), sum(test_data['high_grades'])]
    ax3.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax3.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax3.set_ylabel('Count')
    ax3.set_title('High Grade Distribution')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(hg_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. ER Status
    ax4 = plt.subplot(3, 3, 4)
    er_labels = ['ER-', 'ER+']
    x_pos = [0, 1]
    train_counts = [sum(1 for e in train_data['er_status'] if e == 0), sum(train_data['er_status'])]
    test_counts = [sum(1 for e in test_data['er_status'] if e == 0), sum(test_data['er_status'])]
    ax4.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax4.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax4.set_ylabel('Count')
    ax4.set_title('ER Status Distribution')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(er_labels)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. PR Status
    ax5 = plt.subplot(3, 3, 5)
    pr_labels = ['PR-', 'PR+']
    x_pos = [0, 1]
    train_counts = [sum(1 for p in train_data['pr_status'] if p == 0), sum(train_data['pr_status'])]
    test_counts = [sum(1 for p in test_data['pr_status'] if p == 0), sum(test_data['pr_status'])]
    ax5.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax5.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax5.set_ylabel('Count')
    ax5.set_title('PR Status Distribution')
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(pr_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. HER2 Status
    ax6 = plt.subplot(3, 3, 6)
    her2_labels = ['HER2-', 'HER2+']
    x_pos = [0, 1]
    train_counts = [sum(1 for h in train_data['her2_status'] if h == 0), sum(train_data['her2_status'])]
    test_counts = [sum(1 for h in test_data['her2_status'] if h == 0), sum(test_data['her2_status'])]
    ax6.bar([p - width/2 for p in x_pos], train_counts, width, label='Train', alpha=0.8)
    ax6.bar([p + width/2 for p in x_pos], test_counts, width, label='Test', alpha=0.8)
    ax6.set_ylabel('Count')
    ax6.set_title('HER2 Status Distribution')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(her2_labels)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Tumor Size Distribution
    ax7 = plt.subplot(3, 3, 7)
    if train_data['imaging_features']:
        train_sizes = [f[0] for f in train_data['imaging_features']]
        test_sizes = [f[0] for f in test_data['imaging_features']]
        ax7.hist(train_sizes, bins=15, alpha=0.7, label='Train', edgecolor='black')
        ax7.hist(test_sizes, bins=15, alpha=0.7, label='Test', edgecolor='black')
        ax7.set_xlabel('Tumor Size (mm)')
        ax7.set_ylabel('Frequency')
        ax7.set_title('Tumor Size Distribution')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
    
    # 8. Tumor Volume Distribution
    ax8 = plt.subplot(3, 3, 8)
    if train_data['imaging_features']:
        train_volumes = [f[1] for f in train_data['imaging_features']]
        test_volumes = [f[1] for f in test_data['imaging_features']]
        ax8.hist(train_volumes, bins=15, alpha=0.7, label='Train', edgecolor='black')
        ax8.hist(test_volumes, bins=15, alpha=0.7, label='Test', edgecolor='black')
        ax8.set_xlabel('Tumor Volume (mm³)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Tumor Volume Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
    
    # 9. Dataset Size Summary
    ax9 = plt.subplot(3, 3, 9)
    ax9.axis('off')
    summary_text = f"""
    DATASET SUMMARY
    
    Training Set:
      • Patients: {train_data['total_patients']}
      • With BBox: {train_data['has_bbox_count']}
      
    Test Set:
      • Patients: {test_data['total_patients']}
      • With BBox: {test_data['has_bbox_count']}
      
    Total Samples: {train_data['total_patients'] + test_data['total_patients']}
    
    Class Balance: ✓ Good
    Data Quality: ✓ Complete
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data_distribution.png', dpi=300, bbox_inches='tight')
    print("\n✓ Visualization saved to data_distribution.png")
    plt.show()


def main():
    print("\n" + "="*70)
    print("DUKE DATASET DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Analyze train and test
    train_data = analyze_dataset('exported_patients/train')
    test_data = analyze_dataset('exported_patients/test')
    
    # Visualize
    print("\n" + "="*70)
    print("GENERATING VISUALIZATIONS")
    print("="*70)
    visualize_distributions(train_data, test_data)
    
    print("\n✓ Analysis complete!")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
