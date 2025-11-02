""" 
Brest Cancer Segmentation
----------------------------
Student Names:  Alessandro Anzellini - 2031062, 
                Djibril Coulybaly - 2162985, 
                Clara Gonçalves - 2128459 

Description of File:    Training and evaluation script used with the preprocessed CBIS-DDSM dataset.
"""

# Initialize imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50

import numpy as np
import pandas as pd
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


# ==================== CBIS-DDSM DATASET LOADER ====================

class CBISDDSMDataset(Dataset):
    """PyTorch Dataset for CBIS-DDSM using master_database.csv"""
    def __init__(self, csv_path, split='train', transform=None, feature_cols=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        self.transform = transform

        if feature_cols is None:
            feature_cols = ['breast_density', 'assessment', 'subtlety', 'rank']
        self.feature_cols = feature_cols

        self.scaler = StandardScaler()
        feats = self.df[self.feature_cols].fillna(0).astype(float)
        self.scaler.fit(feats)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Using preprocessed numpy image if available, otherwise load DICOM directly
        preprocessed_path = row.get('preprocessed_path', None)
        image = None
        if preprocessed_path and os.path.exists(preprocessed_path):
            ext = os.path.splitext(preprocessed_path)[1].lower()
            try:
                if ext == '.npy':
                    arr = np.load(preprocessed_path)
                    # If shape is (H, W, 3) or (3, H, W), convert to PIL Image
                    if arr.ndim == 3:
                        if arr.shape[0] == 3:
                            arr = np.transpose(arr, (1, 2, 0))
                        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
                        image = Image.fromarray(arr, mode='RGB')
                    else:
                        arr = ((arr - arr.min()) / (arr.max() - arr.min()) * 255).astype(np.uint8)
                        image = Image.fromarray(arr, mode='L').convert('RGB')
                elif ext in ['.png', '.jpg', '.jpeg']:
                    image = Image.open(preprocessed_path).convert('RGB')
            except Exception as e:
                print(f"Error loading preprocessed file {preprocessed_path}: {e}")
                image = None

        if image is None:
            dcm_path = row['file_path']
            try:
                dcm = pydicom.dcmread(str(dcm_path))
                img = dcm.pixel_array
                if img.min() == img.max():
                    img = np.zeros_like(img)
                else:
                    img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
                if len(img.shape) == 2:
                    image = Image.fromarray(img, mode='L').convert('RGB')
                else:
                    image = Image.new('RGB', (224, 224), color=0)
            except Exception as e:
                print(f"Error loading DICOM {dcm_path}: {e}")
                image = Image.new('RGB', (224, 224), color=0)
        
        if self.transform:
            image = self.transform(image)

        feats = row[self.feature_cols].fillna(0).astype(float).values.reshape(1, -1)
        feats = self.scaler.transform(feats)[0]
        feats = torch.FloatTensor(feats)
        label = 1 if str(row['pathology']).strip().upper() == 'MALIGNANT' else 0
        return image, feats, label, row['patient_id']


# ==================== MULTIMODAL MODEL ====================

class MultimodalBreastCancerModel(nn.Module):
    """Multimodal model combining clinical and imaging features"""
    
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
    
    def __init__(self, model, device, lr=1e-4, weight_decay=1e-4, class_weights=None):
        self.model = model
        self.device = device

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
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
        for images, feats, labels, _ in train_loader:
            images = images.to(self.device)
            feats = feats.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            imaging_feats = torch.zeros((feats.shape[0], 1), device=feats.device)
            outputs = self.model(images, feats, imaging_feats)
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
            for images, feats, labels, _ in val_loader:
                images = images.to(self.device)
                feats = feats.to(self.device)
                labels = labels.to(self.device)
                imaging_feats = torch.zeros((feats.shape[0], 1), device=feats.device)

                outputs = self.model(images, feats, imaging_feats)
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
    print("CBIS-DDSM DATASET STRUCTURE CHECK (from final_database.csv)")
    print("="*60)
    csv_path = Path(__file__).parent.parent / 'final_database.csv'
    if not csv_path.exists():
        print(f"final_database.csv not found at {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"Splits: {df['split'].value_counts().to_dict()}")
    for split in ['train', 'test']:
        sdf = df[df['split'] == split]
        print(f"\n{split.upper()} split: {len(sdf)} samples")
        if len(sdf) == 0:
            continue
        # Show first 3 samples
        for i, row in sdf.head(3).iterrows():
            print(f"  Sample {i}: patient_id={row['patient_id']}, file_type={row['file_type']}, file_path={row['file_path']}")
            # Try to load DICOM and print shape
            try:
                dcm = pydicom.dcmread(str(row['file_path']))
                arr = dcm.pixel_array
                print(f"    - DICOM shape: {arr.shape}, dtype: {arr.dtype}")
            except Exception as e:
                print(f"    - Could not load DICOM: {e}")
            # Print features and label
            feats = {k: row[k] for k in ['breast_density', 'assessment', 'subtlety', 'rank'] if k in row}
            print(f"    - Features: {feats}")
            print(f"    - Pathology: {row['pathology']}")
        if len(sdf) > 3:
            print(f"  ... and {len(sdf) - 3} more samples")
    print("\n" + "="*60)


def main():
    print("CBIS-DDSM Multimodal Classification Pipeline")
    print("="*60)
    
    # Configuration
    EXPORT_DIR = 'data'
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {DEVICE}")

    # Check dataset structure (CSV-based)
    check_dataset_structure(EXPORT_DIR)

    # Compute dataset-specific mean and std from preprocessed images (if available), else from DICOMs
    def compute_mean_std_from_paths(paths, max_samples=500):
        # Global sum/sum-squared method (matches ResNet50.ipynb)
        sum_images = np.zeros(3, dtype=np.float64)
        sum_squared_images = np.zeros(3, dtype=np.float64)
        num_pixels = 0
        n = 0
        for path in paths[:max_samples]:
            ext = os.path.splitext(path)[1].lower()
            try:
                if ext == '.npy':
                    arr = np.load(path).astype(np.float32) / 255.0
                    if arr.ndim == 2:
                        arr = np.stack([arr]*3, axis=-1)
                elif ext in ['.png', '.jpg', '.jpeg']:
                    arr = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
                else:
                    continue
                # arr shape: (H, W, 3)
                sum_images += np.sum(arr, axis=(0, 1))
                sum_squared_images += np.sum(arr ** 2, axis=(0, 1))
                num_pixels += arr.shape[0] * arr.shape[1]
                n += 1
            except Exception as e:
                print(f"Error loading {path}: {e}")
        if n == 0 or num_pixels == 0:
            # Use dataset-specific values from ResNet50.ipynb
            return [0.00237972, 0.0023751, 0.00174674], [0.00088792, 0.00102896, 0.00067923]
        mean = sum_images / num_pixels
        std = np.sqrt((sum_squared_images / num_pixels) - mean ** 2)
        return mean.tolist(), std.tolist()

    # Feature columns (update as needed)
    clinical_feature_cols = ['breast_density', 'assessment', 'subtlety', 'rank']  # Example
    imaging_feature_cols = []  # Add imaging features if available in CSV

    # Load datasets
    print("\n[1/3] Loading datasets...")
    csv_path = Path(__file__).parent.parent / 'final_database.csv'
    train_dataset = CBISDDSMDataset(csv_path, split='train', transform=None, feature_cols=clinical_feature_cols)
    test_dataset = CBISDDSMDataset(csv_path, split='test', transform=None, feature_cols=clinical_feature_cols)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("ERROR: No samples loaded. Check your dataset structure.")
        return

    # Always calculate mean/std from available images (preprocessed if available, else DICOMs)
    image_paths = []
    preproc_paths = [p for p in train_dataset.df.get('preprocessed_path', []) if isinstance(p, str) and os.path.exists(p)]
    if preproc_paths:
        image_paths = preproc_paths
        print(f"[INFO] Found {len(image_paths)} preprocessed images for mean/std calculation.")
    else:
        dicom_paths = [p for p in train_dataset.df.get('file_path', []) if isinstance(p, str) and os.path.exists(p)]
        image_paths = dicom_paths
        print(f"[INFO] No preprocessed images found, using {len(image_paths)} DICOMs for mean/std calculation.")

    if image_paths:
        mean, std = compute_mean_std_from_paths(image_paths)
        print(f"[INFO] Using dataset-specific mean: {mean}, std: {std}")
    else:
        mean, std = [0.00237972, 0.0023751, 0.00174674], [0.00088792, 0.00102896, 0.00067923]
        print(f"[WARN] No images found for mean/std calculation, using ResNet50.ipynb fallback values.")

    # Now define transforms with computed mean/std
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Re-instantiate datasets with transforms
    train_dataset.transform = transform_train
    test_dataset.transform = transform_val

    # Use num_workers=4 for faster data loading (tune as needed for your CPU/RAM)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Compute class weights from training set (as in ResNet50.ipynb)
    train_labels = train_dataset.df['pathology'].map(lambda x: 1 if str(x).strip().upper() == 'MALIGNANT' else 0)
    benign_count = (train_labels == 0).sum()
    malignant_count = (train_labels == 1).sum()
    class_weights = torch.tensor([1.0, benign_count / malignant_count], dtype=torch.float32).to(DEVICE)
    print(f"Class Weights: {class_weights.tolist()}")

    # Model: set clinical_dim and imaging_dim based on feature columns
    clinical_dim = len(clinical_feature_cols)
    imaging_dim = len(imaging_feature_cols) if imaging_feature_cols else 1  # Avoid 0-dim

    print("\n[2/3] Training multimodal model...")
    model = MultimodalBreastCancerModel(
        num_classes=2,
        clinical_dim=clinical_dim,
        imaging_dim=imaging_dim
    ).to(DEVICE)

    trainer = MultimodalTrainer(model, DEVICE, lr=1e-4, class_weights=class_weights)
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