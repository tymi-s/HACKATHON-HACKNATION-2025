#!/usr/bin/env python3
"""
RTG Preprocessor - UNIVERSAL FULL AUGMENTATION
Przetwarza CAŁY zbiór danych (NAUKA), tworząc warianty dla każdego zdjęcia.
"""

import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class RTGAugmentor:
    def __init__(self, source_dir: str, output_dir: str, image_size: tuple = (256, 256)):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        
    def process_all(self):
        # 1. Konfiguracja ścieżek
        print(f"🚀 ROZPOCZYNAM PEŁNE PRZETWARZANIE DANYCH")
        print(f"   Źródło: {self.source_dir}")
        
        if not self.source_dir.exists():
            print(f"❌ BŁĄD: Nie znaleziono folderu źródłowego: {self.source_dir}")
            return

        # 2. Czyszczenie starego folderu processed_data (zaczynamy od zera)
        if self.output_dir.exists():
            print("🧹 Czyszczenie starego folderu processed_data...")
            try:
                shutil.rmtree(self.output_dir)
            except:
                print("⚠️ Nie udało się usunąć folderu automatycznie. Usuń go ręcznie jeśli wystąpi błąd.")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Przetwarzanie obu kategorii
        for category in ['czyste', 'brudne']:
            self.process_category(category)
            
    def process_category(self, category: str):
        target_name = "Good" if category == "czyste" else "Bad"
        source_path = self.source_dir / category
        output_path = self.output_dir / target_name
        
        if not source_path.exists():
            print(f"⚠️ OSTRZEŻENIE: Brak folderu {category} w {self.source_dir}")
            return

        output_path.mkdir(parents=True, exist_ok=True)
        
        # Szukamy podfolderów (struktura oryginalna)
        subdirs = [d for d in source_path.iterdir() if d.is_dir()]
        
        print(f"\n📂 Przetwarzanie kategorii '{category}' -> '{target_name}'...")
        print(f"   Znaleziono {len(subdirs)} obiektów. Generowanie wariantów...")
        
        count = 0
        for subdir in tqdm(subdirs):
            images = list(subdir.glob("*.bmp"))
            
            # Logika parowania (szukamy zwykłego + gęstości)
            gray_img = None
            dens_img = None
            for img in images:
                if "czarno" in img.name.lower() or "hi" in img.name.lower():
                    dens_img = img
                else:
                    gray_img = img
            
            # Jeśli mamy parę, tworzymy hybrydę i warianty
            if gray_img and dens_img:
                base_image = self.create_hybrid(gray_img, dens_img)
                if base_image is None: continue
                
                base_name = subdir.name
                
                # Zapisz oryginał
                cv2.imwrite(str(output_path / f"{base_name}_orig.png"), base_image)
                count += 1
                
                # --- AUGMENTACJA (Tworzymy dodatkowe dane) ---
                
                # 1. Odbicie lustrzane
                cv2.imwrite(str(output_path / f"{base_name}_flip.png"), cv2.flip(base_image, 1))
                count += 1
                
                # 2. Jasność (+20%)
                cv2.imwrite(str(output_path / f"{base_name}_bright.png"), self.adjust_gamma(base_image, 1.2))
                count += 1
                
                # 3. Przyciemnienie (-20%)
                cv2.imwrite(str(output_path / f"{base_name}_dark.png"), self.adjust_gamma(base_image, 0.8))
                count += 1

                # 4. Obrót lekki (+3 stopnie)
                cv2.imwrite(str(output_path / f"{base_name}_rotP3.png"), self.rotate_image(base_image, 3))
                count += 1

                # 5. Zoom (10%)
                cv2.imwrite(str(output_path / f"{base_name}_zoom.png"), self.zoom_image(base_image, 1.1))
                count += 1

        print(f"✅ Zakończono {category}. Łącznie plików w folderze: {count}")

    # --- FUNKCJE POMOCNICZE (Te same co wcześniej) ---
    def create_hybrid(self, gray_path, dens_path):
        img_gray = cv2.imread(str(gray_path), cv2.IMREAD_GRAYSCALE)
        img_dens = cv2.imread(str(dens_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None or img_dens is None: return None
        img_gray = cv2.resize(img_gray, self.image_size)
        img_dens = cv2.resize(img_dens, self.image_size)
        return np.stack([img_gray, img_dens, cv2.addWeighted(img_gray, 0.5, img_dens, 0.5, 0)], axis=-1)

    def rotate_image(self, image, angle):
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def adjust_gamma(self, image, gamma=1.0):
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def zoom_image(self, image, zoom_factor=1.1):
        h, w = image.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        cropped = image[top:top+new_h, left:left+new_w]
        return cv2.resize(cropped, (w, h))

if __name__ == "__main__":
    # PEŁNA ŚCIEŻKA DO DANYCH
    SOURCE = "C:/Users/michm/OneDrive/Pulpit/NAUKA"
    OUTPUT = "processed_data"
    
    # 256x256 to optymalny balans. Jeśli masz bardzo mocny PC, możesz dać 512.
    augmentor = RTGAugmentor(SOURCE, OUTPUT, (256, 256))
    augmentor.process_all()
    print("\n✅ GOTOWE! Teraz uruchom rtg_anomaly_detector.py")