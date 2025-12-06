#!/usr/bin/env python3
"""
RTG Anomaly Detector - GPU / CUDA VERSION
Wersja skonfigurowana do treningu na karcie graficznej.
"""
import os
import warnings
import torch
from pathlib import Path

# --- KONFIGURACJA ŚRODOWISKA ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Kluczowe ustawienie dla GPU na Windows (zapobiega fragmentacji pamięci VRAM)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

# --- NAPRAWA WINDOWS (Monkey Patch) ---
import anomalib.utils.path
anomalib.utils.path.create_versioned_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True) or Path(p)

from anomalib.data import Folder
from anomalib.models import Padim
from anomalib.engine import Engine

def main():
    print("🚀 RTG DETECTOR - GPU/CUDA MODE")
    print("==================================================")

    # 1. SPRAWDZENIE DOSTĘPNOŚCI CUDA
    if not torch.cuda.is_available():
        print("❌ UWAGA: PyTorch nie wykrył karty NVIDIA!")
        print("   Skrypt uruchomi się na CPU, co będzie wolne.")
        print("   Upewnij się, że zainstalowałeś PyTorch z obsługą CUDA.")
        accelerator_type = "cpu"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ Wykryto GPU: {gpu_name}")
        accelerator_type = "gpu"

    # 2. Sprawdzenie danych
    if not Path("processed_data").exists():
        print("❌ BŁĄD: Brak folderu processed_data. Uruchom preprocess_data.py!")
        return

    # 3. Konfiguracja Datasetu
    datamodule = Folder(
        name="rtg_gpu",
        root="processed_data",
        normal_dir="Good",
        abnormal_dir="Bad",
        train_batch_size=8,  # Na GPU możemy dać większy batch (np. 8 lub 16)
        num_workers=2
    )
    datamodule.setup()

    # 4. MODEL
    # Używamy ResNet18 - jest super szybki na GPU i zajmuje mało VRAM.
    # Jeśli masz mocną kartę (np. RTX 3060 lub lepszą z 8GB+ VRAM), 
    # możesz zmienić na "wide_resnet50_2".
    print(f"🧠 Inicjalizacja modelu: Padim (ResNet18)...")
    model = Padim(
        backbone="resnet18", 
        layers=["layer1", "layer2", "layer3"]
    )

    # 5. SILNIK TRENINGOWY (Tu jest zmiana na GPU)
    engine = Engine(
        max_epochs=3,               # Na GPU 3 epoki przelecą błyskawicznie
        default_root_dir="final_results_gpu",
        accelerator=accelerator_type, # <--- TU JEST KLUCZOWA ZMIANA ("gpu")
        devices=1,                  # Użyj 1 karty graficznej
    )

    # 6. Start
    print("\n⏳ Rozpoczynam trening...")
    try:
        engine.fit(model=model, datamodule=datamodule)
        print("✅ Trening zakończony.")
        
        print("\n🧪 Testowanie dokładności...")
        results = engine.test(model=model, datamodule=datamodule)
        
        if results:
            metrics = results[0]
            auroc = 0.0
            for k, v in metrics.items():
                if "AUROC" in k and isinstance(v, (int, float)):
                    auroc = v
            
            print("\n" + "="*40)
            print(f"🎯 WYNIK KOŃCOWY (AUROC): {auroc:.4f}")
            print("="*40)

    except torch.cuda.OutOfMemoryError:
        print("\n❌ BŁĄD: Brak pamięci VRAM na karcie graficznej.")
        print("   Rozwiązanie: Zmniejsz 'train_batch_size' w kodzie do 2 lub 1.")
    except Exception as e:
        print(f"❌ Błąd: {e}")

if __name__ == "__main__":
    main()