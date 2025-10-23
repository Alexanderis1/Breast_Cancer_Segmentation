import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50

import numpy as np
import pandas as pd
import json
import os
from pathlib import Path
from PIL import Image
import pydicom
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ==================== MULTIMODAL DATASET LOADER ====================

class DukeMultimodalDataset(Dataset):
    """Load Duke dataset with MRI images + clinical/imaging features"""
    
    def __init__(self, patients_dir, transform=None, sample_slices=5, 
                 use_clinical=True, use_imaging_features=True):
        self.patients_dir = Path(patients_dir)
        self.transform = transform
        self.sample_slices = sample_slices
        self.use_clinical = use_clinical
        self.use_imaging_features = use_imaging_features
        
        self.samples = []
        self.clinical_scaler = StandardScaler()
        self.imaging_scaler = StandardScaler()
        
        self.load_dataset()
        
    def load_dataset(self):
        """Load all patient data with multimodal features"""
        patient_folders = [d for d in self.patients_dir.iterdir() if d.is_dir()]
        
        clinical_features_list = []
        imaging_features_list = []
        
        for patient_dir in sorted(patient_folders):
            data_file = patient_dir / 'patient_data.json'
            
            if not data_file.exists():
                continue
            
            try:
                with open(data_file) as f:
                    metadata = json.load(f)
                
                # Get MRI files from metadata
                mri_files = metadata.get('mri_files', [])
                dicom_files = []
                
                if not mri_files:
                    # Fallback: look for DICOM files in MRI directory
                    mri_dir = patient_dir / 'MRI_DICOM_sample'
                    if mri_dir.exists():
                        dicom_files = sorted(list(mri_dir.glob('**/*.dcm')))
                else:
                    # Reconstruct paths from metadata
                    for mri_info in mri_files:
                        # Try multiple path formats
                        desc_path = mri_info.get('descriptive_path', '')
                        orig_path = mri_info.get('original_path_and_filename', '')
                        
                        # Try descriptive path first
                        if desc_path:
                            parts = desc_path.split('/')
                            if len(parts) >= 1:
                                # Try finding the DICOM file
                                filename = parts[-1]
                                # Search in MRI_DICOM_sample directory
                                mri_dir = patient_dir / 'MRI_DICOM_sample'
                                if mri_dir.exists():
                                    possible_paths = list(mri_dir.glob(f'**/{filename}'))
                                    if possible_paths:
                                        dicom_files.append(possible_paths[0])
                
                # If still no files found, do a full search
                if not dicom_files:
                    mri_dir = patient_dir / 'MRI_DICOM_sample'
                    if mri_dir.exists():
                        dicom_files = sorted(list(mri_dir.glob('**/*.dcm')))
                
                if not dicom_files:
                    print(f"Warning: No DICOM files found for {patient_dir.name}")
                    continue
                
                # Extract features
                clinical_features = self._extract_clinical_features(metadata)
                imaging_features = self._extract_imaging_features(metadata)
                label = self._extract_label(metadata)
                
                sample_info = {
                    'patient_id': patient_dir.name,
                    'metadata': metadata,
                    'dicom_files': dicom_files,
                    'clinical_features': clinical_features,
                    'imaging_features': imaging_features,
                    'label': label,
                    'patient_dir': patient_dir
                }
                
                self.samples.append(sample_info)
                
                if clinical_features is not None:
                    clinical_features_list.append(clinical_features)
                if imaging_features is not None:
                    imaging_features_list.append(imaging_features)
                
            except Exception as e:
                print(f"Error loading {patient_dir.name}: {e}")
                continue
        
        # Fit scalers
        if clinical_features_list and self.use_clinical:
            self.clinical_scaler.fit(clinical_features_list)
        if imaging_features_list and self.use_imaging_features:
            self.imaging_scaler.fit(imaging_features_list)
        
        print(f"Loaded {len(self.samples)} samples")
    
    def _extract_clinical_features(self, metadata):
        """Extract numerical clinical features"""
        if not self.use_clinical:
            return None
            
        clinical = metadata.get('demographic_clinical', {})
        if not clinical:
            return None
        
        # Key clinical features (numerical only)
        feature_keys = [
            'Days to MRI (From the Date of Diagnosis)',
            'Field Strength (Tesla)',
            'TR (Repetition Time)',
            'TE (Echo Time)',
            'Slice Thickness ',
            'Age at last contact in EMR f/u(days)(from the date of diagnosis) ,last time patient known to be alive, unless age of death is reported(in such case the age of death',
            'Staging(Tumor Size)#[T]',
            'Staging(Nodes)#(Nx replaced by -1)[N]',
            'Staging(Metastasis)#(Mx -replaced by -1)[M]',
            'Tumor Grade',
            'ER',
            'PR',
            'HER2',
            'Menopause (at diagnosis)'
        ]
        
        features = []
        for key in feature_keys:
            val = clinical.get(key, 0)
            if isinstance(val, (int, float)) and not np.isnan(val):
                features.append(float(val))
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_imaging_features(self, metadata):
        """Extract key imaging features"""
        if not self.use_imaging_features:
            return None
            
        imaging = metadata.get('imaging_features', {})
        if not imaging:
            return None
        
        # Key radiomic features (12 features to match)
        feature_keys = [
            'TumorMajorAxisLength_mm',
            'Volume_cu_mm_Tumor',
            'Energy_Tumor',
            'Contrast_Tumor',
            'Homogeneity1_Tumor',
            'breastDensity_T1',
            'breastDensity_PostCon',
            'Max_Enhancement_from_char_curv',
            'Time_to_Peak_from_char_curv',
            'Uptake_rate_from_char_curv',
            'Washout_rate_from_char_curv',
            'Peak_SER_tumor'
        ]
        
        features = []
        for key in feature_keys:
            val = imaging.get(key, 0)
            if isinstance(val, (int, float)) and not np.isnan(val):
                features.append(float(val))
            else:
                features.append(0.0)
        
        return np.array(features, dtype=np.float32)
    
    def _extract_label(self, metadata):
        """Extract label - malignant if annotations exist"""
        annotations = metadata.get('annotations', [])
        
        # Check clinical data for explicit cancer indicators
        clinical = metadata.get('demographic_clinical', {})
        
        # If has annotations or tumor staging > 0, consider malignant
        if annotations and len(annotations) > 0:
            return 1
        
        tumor_stage = clinical.get('Staging(Tumor Size)#[T]', 0)
        if isinstance(tumor_stage, (int, float)) and tumor_stage > 0:
            return 1
        
        # Check if has metastasis info
        metastasis = clinical.get('Metastatic at Presentation (Outside of Lymph Nodes)', 0)
        if metastasis == 1:
            return 1
        
        # Check molecular subtype (if not 0, likely malignant)
        mol_subtype = clinical.get('Mol Subtype', -1)
        if isinstance(mol_subtype, (int, float)) and mol_subtype > 0:
            return 1
            
        return 0
    
    def _load_dicom_image(self, dicom_path):
        """Load DICOM file and extract pixel array"""
        try:
            dcm = pydicom.dcmread(str(dicom_path))
            image = dcm.pixel_array
            
            # Handle different DICOM array shapes
            # Remove singleton dimensions and ensure 2D or 3D
            while len(image.shape) > 2 and image.shape[0] == 1:
                image = image.squeeze(0)
            
            # If still 3D, take the middle slice or first slice
            if len(image.shape) == 3:
                if image.shape[0] < image.shape[-1]:
                    # Likely (slices, H, W)
                    image = image[image.shape[0] // 2]
                else:
                    # Likely (H, W, channels)
                    if image.shape[-1] in [1, 3, 4]:
                        image = image[..., 0]  # Take first channel
                    else:
                        image = image[:, :, image.shape[-1] // 2]
            
            # Ensure 2D at this point
            if len(image.shape) != 2:
                print(f"Warning: Unexpected shape {image.shape} for {dicom_path}")
                return None
            
            # Normalize to 0-255
            if image.min() == image.max():
                image = np.zeros_like(image)
            else:
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
            
            return image
        except Exception as e:
            print(f"Error loading DICOM {dicom_path}: {e}")
            return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        
        # Load DICOM images
        dicom_files = sample['dicom_files']
        num_files = len(dicom_files)
        
        if num_files == 0:
            # Return blank image
            image = Image.new('RGB', (224, 224), color=0)
        else:
            # Sample slices evenly
            indices = np.linspace(0, num_files-1, min(self.sample_slices, num_files), dtype=int)
            
            images = []
            for i in indices:
                img = self._load_dicom_image(dicom_files[i])
                if img is not None and len(img.shape) == 2:  # Ensure 2D
                    images.append(img)
            
            if not images:
                image = Image.new('RGB', (224, 224), color=0)
            else:
                # Average the slices
                stacked = np.stack(images, axis=0)
                avg_slice = stacked.mean(axis=0).astype(np.uint8)
                
                # Ensure 2D before converting to PIL
                if len(avg_slice.shape) > 2:
                    avg_slice = avg_slice.squeeze()
                
                # Convert grayscale to RGB
                if len(avg_slice.shape) == 2:
                    image = Image.fromarray(avg_slice, mode='L').convert('RGB')
                else:
                    # Shouldn't happen but handle just in case
                    image = Image.new('RGB', (224, 224), color=0)
        
        if self.transform:
            image = self.transform(image)
        
        # Get tabular features
        clinical_feat = sample['clinical_features']
        imaging_feat = sample['imaging_features']
        
        # Scale features
        if clinical_feat is not None and self.use_clinical:
            clinical_feat = self.clinical_scaler.transform([clinical_feat])[0]
            clinical_feat = torch.FloatTensor(clinical_feat)
        else:
            clinical_feat = torch.zeros(14)  # Updated to 14
        
        if imaging_feat is not None and self.use_imaging_features:
            imaging_feat = self.imaging_scaler.transform([imaging_feat])[0]
            imaging_feat = torch.FloatTensor(imaging_feat)
        else:
            imaging_feat = torch.zeros(12)  # Updated to 12
        
        return image, clinical_feat, imaging_feat, label, sample['patient_id']


# ==================== MULTIMODAL MODEL ====================

class MultimodalBreastCancerModel(nn.Module):
    """Multimodal model combining MRI images + clinical + imaging features"""
    
    def __init__(self, num_classes=2, clinical_dim=14, imaging_dim=12, dropout_rate=0.3):
        super().__init__()
        
        # Image branch (ResNet50)
        self.image_backbone = resnet50(weights='IMAGENET1K_V1')
        self.image_backbone.fc = nn.Identity()
        image_feat_dim = 2048
        
        # Clinical features branch
        self.clinical_branch = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Imaging features branch
        self.imaging_branch = nn.Sequential(
            nn.Linear(imaging_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        
        # Fusion classifier
        fusion_dim = image_feat_dim + 32 + 32
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, image, clinical_feat, imaging_feat):
        # Extract image features
        img_feat = self.image_backbone(image)
        if len(img_feat.shape) > 2:
            img_feat = F.adaptive_avg_pool2d(img_feat, (1, 1)).flatten(1)
        
        # Extract clinical features
        clin_feat = self.clinical_branch(clinical_feat)
        
        # Extract imaging features
        imag_feat = self.imaging_branch(imaging_feat)
        
        # Concatenate all features
        fused = torch.cat([img_feat, clin_feat, imag_feat], dim=1)
        
        # Classification
        output = self.classifier(fused)
        return output


# ==================== TRAINING & EVALUATION ====================

class MultimodalTrainer:
    """Training loop manager for multimodal model"""
    
    def __init__(self, model, device, lr=1e-4, weight_decay=1e-4):
        self.model = model
        self.device = device
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        self.history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        self.best_auc = 0.0
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for images, clinical, imaging, labels, _ in train_loader:
            images = images.to(self.device)
            clinical = clinical.to(self.device)
            imaging = imaging.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images, clinical, imaging)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, clinical, imaging, labels, _ in val_loader:
                images = images.to(self.device)
                clinical = clinical.to(self.device)
                imaging = imaging.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images, clinical, imaging)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        
        # Handle case where only one class is present
        unique_labels = np.unique(all_labels)
        if len(unique_labels) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.5
            print(f"Warning: Only class {unique_labels[0]} present in validation set")
        
        return val_loss, auc, np.array(all_labels), np.array(all_preds), np.array(all_probs)
    
    def train(self, train_loader, val_loader, num_epochs=20, patience=5):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_auc, _, _, _ = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_auc'].append(val_auc)
            
            self.scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val AUC: {val_auc:.4f}")
            
            if val_loss < best_loss:
                best_loss = val_loss
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), 'best_multimodal_model.pth')
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        self.model.load_state_dict(torch.load('best_multimodal_model.pth'))


def plot_results(history, labels, preds, probs, save_path='multimodal_results.png'):
    """Plot training history and evaluation metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val', marker='s')
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # AUC
    axes[0, 1].plot(history['val_auc'], label='Val AUC', marker='o', color='green')
    axes[0, 1].set_title('AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    axes[1, 0].set_title('Confusion Matrix')
    
    # ROC Curve
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)
        axes[1, 1].plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
        axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 1].set_title('ROC Curve')
        axes[1, 1].set_xlabel('FPR')
        axes[1, 1].set_ylabel('TPR')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC curve unavailable\n(single class)', 
                       ha='center', va='center')
        axes[1, 1].set_title('ROC Curve')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Results saved to {save_path}")
    plt.close()


# ==================== MAIN PIPELINE ====================

def check_dataset_structure(export_dir):
    """Check and print dataset structure for debugging"""
    print("\n" + "="*60)
    print("DATASET STRUCTURE CHECK")
    print("="*60)
    
    export_path = Path(export_dir)
    
    for split in ['train', 'test']:
        split_dir = export_path / split
        if not split_dir.exists():
            print(f"\n{split.upper()}: NOT FOUND")
            continue
        
        print(f"\n{split.upper()} Split:")
        patient_folders = [d for d in split_dir.iterdir() if d.is_dir()]
        print(f"  Total patient folders: {len(patient_folders)}")
        
        for patient_dir in sorted(patient_folders)[:3]:  # Show first 3
            print(f"\n  Patient: {patient_dir.name}")
            
            # Check patient_data.json
            json_file = patient_dir / 'patient_data.json'
            print(f"    - patient_data.json: {json_file.exists()}")
            
            if json_file.exists():
                with open(json_file) as f:
                    data = json.load(f)
                print(f"    - Has clinical data: {bool(data.get('demographic_clinical'))}")
                print(f"    - Has imaging features: {bool(data.get('imaging_features'))}")
                print(f"    - Has annotations: {len(data.get('annotations', []))} annotations")
                print(f"    - MRI files in metadata: {len(data.get('mri_files', []))}")
            
            # Check MRI directory
            mri_dir = patient_dir / 'MRI_DICOM_sample'
            if mri_dir.exists():
                dcm_files = list(mri_dir.glob('**/*.dcm'))
                print(f"    - MRI_DICOM_sample: {len(dcm_files)} DICOM files found")
                if dcm_files and len(dcm_files) <= 3:
                    for dcm in dcm_files:
                        print(f"      • {dcm.relative_to(mri_dir)}")
            else:
                print(f"    - MRI_DICOM_sample: NOT FOUND")
            
            # Check segmentation directory
            seg_dir = patient_dir / 'Segmentation_Masks_NRRD'
            if seg_dir.exists():
                nrrd_files = list(seg_dir.glob('*.nrrd'))
                print(f"    - Segmentation_Masks_NRRD: {len(nrrd_files)} NRRD files")
            
        if len(patient_folders) > 3:
            print(f"\n  ... and {len(patient_folders) - 3} more patients")
    
    print("\n" + "="*60)


def main():
    print("Duke Breast Cancer Multimodal Classification Pipeline")
    print("="*60)
    
    # Configuration
    EXPORT_DIR = 'data'
    BATCH_SIZE = 4
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    
    # Check dataset structure first
    check_dataset_structure(EXPORT_DIR)
    
    # Define transforms
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    print("\n[1/3] Loading datasets...")
    train_dataset = DukeMultimodalDataset(
        Path(EXPORT_DIR) / 'train',
        transform=transform_train,
        use_clinical=True,
        use_imaging_features=True
    )
    
    test_dataset = DukeMultimodalDataset(
        Path(EXPORT_DIR) / 'test',
        transform=transform_val,
        use_clinical=True,
        use_imaging_features=True
    )
    
    # Copy scalers from train to test
    test_dataset.clinical_scaler = train_dataset.clinical_scaler
    test_dataset.imaging_scaler = train_dataset.imaging_scaler
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("ERROR: No samples loaded. Check your dataset structure.")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    print("\n[2/3] Training multimodal model...")
    model = MultimodalBreastCancerModel(
        num_classes=2,
        clinical_dim=14,  # Updated to 14
        imaging_dim=12    # Updated to 12
    ).to(DEVICE)
    
    trainer = MultimodalTrainer(model, DEVICE, lr=1e-4)
    trainer.train(train_loader, test_loader, num_epochs=NUM_EPOCHS, patience=5)
    
    # Evaluate
    print("\n[3/3] Evaluating model...")
    val_loss, val_auc, labels, preds, probs = trainer.validate(test_loader)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"AUC: {val_auc:.4f}")
    print(f"Accuracy: {(preds == labels).mean():.4f}")
    
    if len(np.unique(labels)) > 1:
        print("\nClassification Report:")
        print(classification_report(labels, preds, target_names=['Benign', 'Malignant']))
    else:
        print(f"\nWarning: Only class {np.unique(labels)[0]} in test set")
    
    plot_results(trainer.history, labels, preds, probs)


if __name__ == '__main__':
    main()