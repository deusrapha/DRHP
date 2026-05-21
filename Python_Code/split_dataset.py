import os
import random
import shutil
from pathlib import Path

# Paths
SOURCE_DIR = Path(r"C:\New folder\Local Disk\Masters MCS\SEM_II\Computer Vision\2026-19\labbled\labbled")
DEST_DIR = Path(r"C:\New folder\Local Disk\Masters MCS\SEM_II\Computer Vision\2026-19\dataset")

# Get all images
images = list(SOURCE_DIR.glob("*.jpg"))
random.seed(42)  # For reproducibility
random.shuffle(images)

# Calculate splits (70% train, 20% val, 10% test)
total = len(images)
train_split = int(total * 0.7)
val_split = int(total * 0.9)

train_images = images[:train_split]
val_images = images[train_split:val_split]
test_images = images[val_split:]

def copy_files(image_list, split_name):
    img_dest = DEST_DIR / split_name / "images"
    lbl_dest = DEST_DIR / split_name / "labels"
    
    img_dest.mkdir(parents=True, exist_ok=True)
    lbl_dest.mkdir(parents=True, exist_ok=True)
    
    for img_path in image_list:
        # Copy image
        shutil.copy2(img_path, img_dest / img_path.name)
        
        # Copy label
        lbl_path = img_path.with_suffix(".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, lbl_dest / lbl_path.name)

print("Copying training data...")
copy_files(train_images, "train")
print("Copying validation data...")
copy_files(val_images, "val")
print("Copying test data...")
copy_files(test_images, "test")

print(f"Dataset split complete! Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
