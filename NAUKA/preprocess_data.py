#!/usr/bin/env python3
"""
Data Preprocessing for RTG Images
Handles dual-channel images: grayscale + density (blue) images
"""

import os
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm


class RTGPreprocessor:
    """
    Preprocesses RTG images for anomaly detection.
    
    Combines grayscale and density images into multi-channel format.
    """
    
    def __init__(self, source_dir: str, output_dir: str, image_size: tuple = (512, 512)):
        """
        Args:
            source_dir: Directory with raw data (czyste, brudne folders)
            output_dir: Where to save preprocessed images
            image_size: Target size for images
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
    def process_all(self):
        """Process all images in czyste and brudne folders"""
        for category in ['czyste', 'brudne']:
            print(f"\n📂 Processing {category} images...")
            self.process_category(category)
            
    def process_category(self, category: str):
        """
        Process images in a category (czyste or brudne).
        
        Expected input structure:
        source_dir/czyste/
        ├── IMG_001/
        │   ├── IMG_001.bmp          # Grayscale RTG
        │   └── IMG_001_czarno.bmp   # Density (blue) image
        └── IMG_002/
            └── ...
        """
        category_path = self.source_dir / category
        output_category = self.output_dir / category
        output_category.mkdir(parents=True, exist_ok=True)
        
        # Get all subdirectories
        subdirs = [d for d in category_path.iterdir() if d.is_dir()]
        
        for subdir in tqdm(subdirs, desc=f"Processing {category}"):
            # Find the two image files
            images = list(subdir.glob("*.bmp"))
            
            # Separate grayscale and density images
            grayscale_img = None
            density_img = None
            
            for img_path in images:
                if "czarno" in img_path.name.lower():
                    density_img = img_path
                else:
                    grayscale_img = img_path
            
            if grayscale_img is None or density_img is None:
                print(f"⚠️  Skipping {subdir.name} - missing images")
                continue
            
            # Process and combine images
            self.process_image_pair(
                grayscale_path=grayscale_img,
                density_path=density_img,
                output_path=output_category / f"{subdir.name}.png",
            )
    
    def process_image_pair(self, grayscale_path: Path, density_path: Path, output_path: Path):
        """
        Process a pair of images (grayscale + density).
        
        Strategies:
        1. Stack as 2-channel image
        2. Combine into RGB (grayscale, density, weighted_avg)
        3. Side-by-side concatenation
        """
        # Load images
        gray = cv2.imread(str(grayscale_path), cv2.IMREAD_GRAYSCALE)
        density = cv2.imread(str(density_path), cv2.IMREAD_GRAYSCALE)
        
        # Resize to target size
        gray = cv2.resize(gray, self.image_size)
        density = cv2.resize(density, self.image_size)
        
        # Strategy 1: Stack as 2-channel (save as 3-channel RGB for compatibility)
        # We use grayscale in R channel, density in G channel, and combination in B
        combined = np.stack([
            gray,                           # R channel: grayscale
            density,                        # G channel: density
            (gray * 0.7 + density * 0.3).astype(np.uint8)  # B channel: weighted combination
        ], axis=-1)
        
        # Save
        cv2.imwrite(str(output_path), combined)
    
    def create_anomalib_structure(self):
        """
        Create proper Anomalib folder structure:
        
        output_dir/
        ├── normal/
        │   └── czyste/
        │       ├── IMG_001.png
        │       └── IMG_002.png
        └── abnormal/
            └── brudne/
                ├── IMG_003.png
                └── IMG_004.png
        """
        # Create structure
        normal_dir = self.output_dir / "normal" / "czyste"
        abnormal_dir = self.output_dir / "abnormal" / "brudne"
        
        normal_dir.mkdir(parents=True, exist_ok=True)
        abnormal_dir.mkdir(parents=True, exist_ok=True)
        
        # Move processed images
        if (self.output_dir / "czyste").exists():
            for img in (self.output_dir / "czyste").glob("*.png"):
                img.rename(normal_dir / img.name)
            (self.output_dir / "czyste").rmdir()
        
        if (self.output_dir / "brudne").exists():
            for img in (self.output_dir / "brudne").glob("*.png"):
                img.rename(abnormal_dir / img.name)
            (self.output_dir / "brudne").rmdir()
        
        print(f"\n✅ Created Anomalib structure in {self.output_dir}")


def explore_data(data_dir: str):
    """
    Explore the raw data structure and show statistics.
    """
    data_path = Path(data_dir)
    
    print("\n" + "=" * 70)
    print("DATA EXPLORATION")
    print("=" * 70)
    
    for category in ['czyste', 'brudne']:
        category_path = data_path / category
        if not category_path.exists():
            print(f"\n⚠️  {category} directory not found!")
            continue
            
        subdirs = [d for d in category_path.iterdir() if d.is_dir()]
        print(f"\n📊 {category.upper()}:")
        print(f"   Total subdirectories: {len(subdirs)}")
        
        # Sample first directory to understand structure
        if subdirs:
            sample = subdirs[0]
            images = list(sample.glob("*.bmp"))
            print(f"   Sample directory: {sample.name}")
            print(f"   Images per directory: {len(images)}")
            
            # Check image sizes
            if images:
                img = Image.open(images[0])
                print(f"   Image size: {img.size} (WxH)")
                print(f"   Image mode: {img.mode}")
                file_size_mb = images[0].stat().st_size / (1024 * 1024)
                print(f"   File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "../NAUKA"          # <-- ZMIEŃ NA SWOJĄ ŚCIEŻKĘ!
    OUTPUT_DIR = "./processed_data"  # Zostaje tak
    IMAGE_SIZE = (512, 512)          # Możesz zmienić na (256, 256) jeśli za wolno
    
    print("=" * 70)
    print("RTG DATA PREPROCESSING")
    print("=" * 70)
    
    # First, explore the data
    explore_data(SOURCE_DIR)
    
    # Create preprocessor
    preprocessor = RTGPreprocessor(
        source_dir=SOURCE_DIR,
        output_dir=OUTPUT_DIR,
        image_size=IMAGE_SIZE,
    )
    
    # Process all images
    print("\n🔄 Starting preprocessing...")
    preprocessor.process_all()
    
    # Create Anomalib-compatible structure
    preprocessor.create_anomalib_structure()
    
    print("\n✅ Preprocessing complete!")
    print(f"   Processed images saved to: {OUTPUT_DIR}")
    print("\nNext step:")
    print(f"   python rtg_anomaly_detector.py")