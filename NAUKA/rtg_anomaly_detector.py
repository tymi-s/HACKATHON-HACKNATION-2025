#!/usr/bin/env python3
"""
RTG Contraband Detection - MEMORY OPTIMIZED
Fix for large images + Windows symlink patch
"""

import os
import sys
from pathlib import Path
import warnings

# --- KONFIGURACJA ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")

# --- MONKEY PATCH ---
import anomalib.utils.path

def fake_create_versioned_dir(root_dir):
    root_dir = Path(root_dir)
    root_dir.mkdir(parents=True, exist_ok=True)
    return root_dir

print("🔧 Applying Windows symlink patch...")
anomalib.utils.path.create_versioned_dir = fake_create_versioned_dir
print("✅ Patch applied!")

# --- IMPORTY ---
from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine


class RTGAnomalyDetector:
    def __init__(self, data_root: str, backbone: str = "resnet18"):
        self.data_root = Path(data_root)
        
        # ZMIANA: resnet18 zamiast wide_resnet50_2 (mniej pamięci!)
        # ZMIANA: n_features=100 zamiast 550 (znacznie mniej RAM!)
        self.model = Padim(
            backbone=backbone,
            layers=["layer1", "layer2", "layer3"],
            n_features=100,  # KLUCZOWE: 100 zamiast 550!
        )
        
        self.engine = None
        
    def prepare_dataset(self):
        print(f"📁 Loading data from: {self.data_root}")
        
        self.datamodule = Folder(
            name="rtg_vehicles",
            root=self.data_root,
            normal_dir="normal/czyste",
            abnormal_dir="abnormal/brudne",
        )
        
        self.datamodule.setup()
        
        train_size = len(self.datamodule.train_dataloader().dataset)
        test_size = len(self.datamodule.test_dataloader().dataset)
        
        print(f"✅ Dataset prepared:")
        print(f"   Training: {train_size} | Test: {test_size}")
        
        return self.datamodule
    
    def train(self, max_epochs: int = 1):
        print(f"\n🚀 Training PADIM (memory optimized)...")
        
        results_dir = Path("./final_results")
        
        self.engine = Engine(
            accelerator="cpu",  # Force CPU
            devices=1,
            max_epochs=max_epochs,
            default_root_dir=str(results_dir),
        )
        
        print("⏳ Training...")
        self.engine.fit(model=self.model, datamodule=self.datamodule)
        print("✅ Done!")
        
    def test(self):
        print("\n🧪 Testing...")
        results = self.engine.test(model=self.model, datamodule=self.datamodule)
        
        print("\n📊 RESULTS:")
        if results and len(results) > 0:
            for key, value in results[0].items():
                if isinstance(value, (int, float)):
                    print(f"   {key}: {value:.4f}")
        
        return results


def main():
    DATA_ROOT = "./processed_data"
    
    print("=" * 70)
    print("RTG ANOMALY DETECTOR - MEMORY OPTIMIZED")
    print("=" * 70)
    print("\n⚠️  Using ResNet18 + reduced features for memory efficiency")
    print("    Your images are HUGE (2164x1250), need to optimize!\n")
    
    if not Path(DATA_ROOT).exists():
        print("❌ ERROR: Data not found")
        return

    detector = RTGAnomalyDetector(data_root=DATA_ROOT, backbone="resnet18")
    detector.prepare_dataset()
    detector.train(max_epochs=1)
    detector.test()
    
    print("\n✅ DONE!")

if __name__ == "__main__":
    main()