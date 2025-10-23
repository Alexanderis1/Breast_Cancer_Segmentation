import nrrd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from datetime import datetime

class NRRDViewer:
    """Viewer for NRRD segmentation masks from Duke dataset - saves all outputs"""
    
    def __init__(self, export_dir=None, output_dir='nrrd_analysis'):
        if export_dir is None:
            possible_paths = [
                'exported_10_patients',
                '/mnt/c/Users/35193/Desktop/Sapienza/1 year/Second Semester/Medical Robotics/Project/dataset/duke_dataset/exported_10_patients',
                Path.cwd() / 'exported_10_patients'
            ]
            
            for path in possible_paths:
                if Path(path).exists():
                    export_dir = path
                    break
            
            if export_dir is None:
                raise FileNotFoundError(
                    "Could not find exported_10_patients directory. "
                    "Please provide the path explicitly."
                )
        
        self.export_dir = Path(export_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / 'images'
        self.images_dir.mkdir(exist_ok=True)
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(exist_ok=True)
        
        self.nrrd_files = []
        self.analysis_data = []
        self.find_nrrd_files()
    
    def find_nrrd_files(self):
        """Find all NRRD files in export directory"""
        print(f"Searching for NRRD files in: {self.export_dir}")
        
        if not self.export_dir.exists():
            print(f"ERROR: Directory does not exist: {self.export_dir}")
            return
        
        for root, dirs, files in os.walk(str(self.export_dir)):
            for file in files:
                if file.endswith('.nrrd') or file.endswith('.seg.nrrd'):
                    full_path = Path(root) / file
                    self.nrrd_files.append(full_path)
        
        print(f"Found {len(self.nrrd_files)} NRRD files\n")
    
    def list_nrrd_files(self):
        """Save list of all NRRD files to text file"""
        output_file = self.data_dir / 'nrrd_file_list.txt'
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write("NRRD FILES IN EXPORT DIRECTORY\n")
            f.write("="*70 + "\n\n")
            
            if not self.nrrd_files:
                f.write("No NRRD files found\n")
                return
            
            for idx, filepath in enumerate(sorted(self.nrrd_files), 1):
                relative_path = filepath.relative_to(self.export_dir)
                f.write(f"{idx}. {relative_path}\n")
        
        print(f"Saved file list to: {output_file}")
    
    def get_nrrd_info(self, filepath):
        """Load NRRD file and extract information"""
        try:
            data, header = nrrd.read(str(filepath))
            
            info = {
                'filepath': str(filepath),
                'filename': filepath.name,
                'shape': data.shape,
                'dtype': str(data.dtype),
                'min': float(np.min(data)),
                'max': float(np.max(data)),
                'mean': float(np.mean(data)),
                'std': float(np.std(data)),
                'unique_values': sorted([float(v) for v in np.unique(data)]),
                'num_unique': len(np.unique(data)),
                'header_keys': list(header.keys())
            }
            return data, header, info
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None, None, None
    
    def inspect_patient_nrrd(self, patient_id=None, split='train'):
        """Inspect NRRD files for a specific patient and save report"""
        output_file = self.data_dir / f'patient_nrrd_inspection_{split}.txt'
        
        with open(output_file, 'w') as f:
            f.write("="*70 + "\n")
            f.write(f"INSPECTING NRRD FILES - {split.upper()}\n")
            f.write("="*70 + "\n\n")
            
            patient_dir = self.export_dir / split
            patient_folders = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
            
            if not patient_folders:
                f.write(f"No patients found in {split} directory\n")
                return
            
            target_dir = patient_folders[0] if patient_id is None else patient_dir / f'Patient_{patient_id}'
            
            if not target_dir.exists():
                f.write(f"Patient directory not found: {target_dir}\n")
                return
            
            f.write(f"Patient: {target_dir.name}\n\n")
            
            seg_dir = target_dir / 'Segmentation_Masks_NRRD'
            if not seg_dir.exists():
                f.write("No segmentation masks directory found\n")
                return
            
            nrrd_files = sorted(list(seg_dir.glob('*.nrrd')) + list(seg_dir.glob('*.seg.nrrd')))
            
            if not nrrd_files:
                f.write("No NRRD files found in this patient's directory\n")
                return
            
            f.write(f"Found {len(nrrd_files)} NRRD files:\n\n")
            
            for idx, filepath in enumerate(nrrd_files, 1):
                data, header, info = self.get_nrrd_info(filepath)
                
                if data is not None:
                    f.write(f"{idx}. {filepath.name}\n")
                    f.write(f"   Shape: {info['shape']}\n")
                    f.write(f"   Data type: {info['dtype']}\n")
                    f.write(f"   Value range: [{info['min']:.1f}, {info['max']:.1f}]\n")
                    f.write(f"   Mean: {info['mean']:.4f}, Std: {info['std']:.4f}\n")
                    f.write(f"   Unique values: {info['unique_values']}\n")
                    f.write(f"   Number of unique classes: {info['num_unique']}\n\n")
                    
                    self.analysis_data.append(info)
        
        print(f"Saved inspection report to: {output_file}")
    
    def visualize_nrrd_slices(self, filepath, num_slices=5):
        """Visualize selected slices from a 3D NRRD file and save as image"""
        data, header, info = self.get_nrrd_info(filepath)
        
        if data is None:
            return
        
        filename_base = filepath.stem.replace('.seg', '')
        output_path = self.images_dir / f'{filename_base}_slices.png'
        
        print(f"Visualizing: {filepath.name}")
        print(f"  Volume shape: {info['shape']}")
        print(f"  Unique values (labels): {info['unique_values']}")
        
        z_axis = len(info['shape']) - 1
        num_z_slices = info['shape'][z_axis]
        
        slice_indices = np.linspace(0, num_z_slices - 1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(1, num_slices, figsize=(16, 3))
        if num_slices == 1:
            axes = [axes]
        
        fig.suptitle(f"{filepath.name}\nShape: {info['shape']}, Labels: {info['unique_values']}", 
                     fontsize=12, fontweight='bold')
        
        for ax_idx, slice_idx in enumerate(slice_indices):
            if z_axis == 0:
                slice_data = data[slice_idx, :, :]
            elif z_axis == 1:
                slice_data = data[:, slice_idx, :]
            else:
                slice_data = data[:, :, slice_idx]
            
            im = axes[ax_idx].imshow(slice_data, cmap='viridis', origin='lower')
            axes[ax_idx].set_title(f"Slice {slice_idx}", fontsize=10)
            axes[ax_idx].axis('off')
            
            cbar = plt.colorbar(im, ax=axes[ax_idx], fraction=0.046, pad=0.04)
            cbar.set_label('Label', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to: {output_path}\n")
    
    def visualize_patient_nrrd(self, patient_id=None, split='train', num_slices=5):
        """Visualize all NRRD masks for a patient"""
        patient_dir = self.export_dir / split
        patient_folders = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        
        if not patient_folders:
            print("No patients found")
            return
        
        patient_folder = patient_folders[0] if patient_id is None else patient_dir / f'Patient_{patient_id}'
        seg_dir = patient_folder / 'Segmentation_Masks_NRRD'
        
        if not seg_dir.exists():
            print("No segmentation directory found")
            return
        
        nrrd_files = sorted(list(seg_dir.glob('*.nrrd')) + list(seg_dir.glob('*.seg.nrrd')))
        
        print(f"Visualizing NRRD files for: {patient_folder.name}")
        print(f"Found {len(nrrd_files)} masks\n")
        
        for filepath in nrrd_files:
            self.visualize_nrrd_slices(filepath, num_slices=num_slices)
    
    def create_comparison_figure(self, patient_id=None, split='train'):
        """Create a comparison figure for all NRRD masks in a patient and save"""
        patient_dir = self.export_dir / split
        patient_folders = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        
        if not patient_folders:
            print("No patients found")
            return
        
        target_dir = patient_folders[0] if patient_id is None else patient_dir / f'Patient_{patient_id}'
        seg_dir = target_dir / 'Segmentation_Masks_NRRD'
        
        if not seg_dir.exists():
            print("No segmentation directory")
            return
        
        nrrd_files = sorted(list(seg_dir.glob('*.nrrd')) + list(seg_dir.glob('*.seg.nrrd')))
        
        if not nrrd_files:
            print("No NRRD files found")
            return
        
        output_path = self.images_dir / f'{target_dir.name}_comparison.png'
        
        fig, axes = plt.subplots(len(nrrd_files), 5, figsize=(18, 4*len(nrrd_files)))
        if len(nrrd_files) == 1:
            axes = [axes]
        
        fig.suptitle(f"All Segmentation Masks - {target_dir.name}", 
                     fontsize=16, fontweight='bold')
        
        for file_idx, filepath in enumerate(nrrd_files):
            data, header, info = self.get_nrrd_info(filepath)
            
            if data is None:
                continue
            
            z_axis = len(info['shape']) - 1
            num_z = info['shape'][z_axis]
            slice_indices = np.linspace(0, num_z - 1, 5, dtype=int)
            
            for ax_idx, slice_idx in enumerate(slice_indices):
                if z_axis == 0:
                    slice_data = data[slice_idx, :, :]
                elif z_axis == 1:
                    slice_data = data[:, slice_idx, :]
                else:
                    slice_data = data[:, :, slice_idx]
                
                ax = axes[file_idx][ax_idx] if len(nrrd_files) > 1 else axes[ax_idx]
                im = ax.imshow(slice_data, cmap='viridis', origin='lower')
                
                if ax_idx == 0:
                    ax.set_ylabel(filepath.name, fontsize=10, fontweight='bold')
                ax.set_title(f"Slice {slice_idx}", fontsize=9)
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved comparison figure to: {output_path}\n")
    
    def save_summary_json(self):
        """Save all analysis data to JSON file"""
        output_file = self.data_dir / 'nrrd_analysis_summary.json'
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'export_dir': str(self.export_dir),
            'total_nrrd_files': len(self.nrrd_files),
            'analysis_count': len(self.analysis_data),
            'files_analyzed': self.analysis_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved JSON summary to: {output_file}")


def main():
    print("\n" + "="*70)
    print("NRRD SEGMENTATION MASK VIEWER - BATCH EXPORT")
    print("="*70 + "\n")
    
    try:
        viewer = NRRDViewer()
        print(f"Found dataset at: {viewer.export_dir}\n")
        print(f"Output directory: {viewer.output_dir}\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nUsage: python visualize_dataset.py")
        return
    
    # Generate all visualizations and reports
    print("Generating file list...")
    viewer.list_nrrd_files()
    
    print("\nInspecting patient NRRD files...")
    viewer.inspect_patient_nrrd(split='train')
    viewer.inspect_patient_nrrd(split='test')
    
    print("Creating slice visualizations...")
    viewer.visualize_patient_nrrd(split='train', num_slices=5)
    viewer.visualize_patient_nrrd(split='test', num_slices=5)
    
    print("Creating comparison figures...")
    viewer.create_comparison_figure(split='train')
    viewer.create_comparison_figure(split='test')
    
    print("Saving JSON summary...")
    viewer.save_summary_json()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"All outputs saved to: {viewer.output_dir}")
    print(f"  Images: {viewer.images_dir}")
    print(f"  Data: {viewer.data_dir}")


if __name__ == '__main__':
    main()