from pathlib import Path

SOURCE_DIR = Path(r"C:\New folder\Local Disk\Masters MCS\SEM_II\Computer Vision\2026-19\labbled\labbled")

images = list(SOURCE_DIR.glob("*.jpg"))

for image_path in images:
    label_path = image_path.with_suffix(".txt")

    # class_id x_center y_center width height
    # This creates one bounding box covering the whole image
    label_path.write_text("0 0.5 0.5 1.0 1.0\n")

print(f"Created {len(images)} YOLO label files.")
