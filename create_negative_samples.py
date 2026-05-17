"""
Generate synthetic negative samples by heavily augmenting Youssef's images.
This creates a 'negative' class so the model learns to distinguish Youssef from others.
"""

import cv2
import numpy as np
from pathlib import Path
import random

DATASET_DIR = Path("dataset")
NEGATIVE_DIR = DATASET_DIR / "negative"
YOUSSEF_DIR = DATASET_DIR / "Youssef"

def create_negative_sample(image_path, output_path, augmentation_level):
    """Apply heavy augmentation to create a negative sample."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    
    h, w = img.shape[:2]
    
    # Heavy augmentation to make it look different
    if augmentation_level % 5 == 0:
        # Extreme color shift
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + random.randint(80, 160)) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.3, 0.7), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    elif augmentation_level % 5 == 1:
        # Flip and blur
        img = cv2.flip(img, random.choice([0, 1]))
        img = cv2.GaussianBlur(img, (15, 15), 0)
    
    elif augmentation_level % 5 == 2:
        # Rotation + tilt
        angle = random.randint(-45, 45)
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    elif augmentation_level % 5 == 3:
        # Zoom in/out + shift
        scale = random.uniform(0.6, 1.4)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        if scale < 1:
            # Pad if zoomed out
            pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
            img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, cv2.BORDER_REFLECT)
        else:
            # Crop if zoomed in
            crop_h, crop_w = (new_h - h) // 2, (new_w - w) // 2
            img = img[crop_h:crop_h+h, crop_w:crop_w+w]
    
    else:
        # Brightness/contrast + noise
        alpha = random.uniform(0.5, 1.5)
        beta = random.randint(-50, 50)
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
    
    cv2.imwrite(str(output_path), img)
    return True

def main():
    # Create negative directory
    NEGATIVE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get all Youssef images
    youssef_images = list(YOUSSEF_DIR.glob("*.jpg")) + list(YOUSSEF_DIR.glob("*.png"))
    
    if not youssef_images:
        print(f"❌ No images found in {YOUSSEF_DIR}")
        return
    
    print(f"📸 Found {len(youssef_images)} Youssef images")
    print(f"🔄 Generating 50 synthetic negative samples...")
    
    count = 0
    for i in range(50):
        source_img = random.choice(youssef_images)
        output_path = NEGATIVE_DIR / f"negative_{i:03d}.jpg"
        
        if create_negative_sample(source_img, output_path, i):
            count += 1
        
        if (i + 1) % 10 == 0:
            print(f"  ✓ Generated {i + 1}/50 samples")
    
    print(f"✅ Created {count} negative samples in {NEGATIVE_DIR}")
    print(f"📂 Dataset now has 2 classes: 'Youssef' and 'negative'")
    print(f"\nNext: Run 'python train_model.py --epochs 20' to retrain with 2 classes")

if __name__ == "__main__":
    main()
