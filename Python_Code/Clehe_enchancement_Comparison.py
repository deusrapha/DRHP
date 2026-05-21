import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Change this to one clear herbal plant image
image_path = Path(r"C:\New folder\Local Disk\Masters MCS\SEM_II\Computer Vision\2026-19\labbled\labbled\clip (10).jpg")

# Output folder
output_dir = Path(r"C:\New folder\Local Disk\Masters MCS\SEM_II\Computer Vision\2026-19\sample_results")
output_dir.mkdir(exist_ok=True)

# Read image
img_bgr = cv2.imread(str(image_path))
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Convert to LAB
lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

# Apply CLAHE to L channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_clahe = clahe.apply(l)

# Merge channels and convert back to RGB
lab_clahe = cv2.merge((l_clahe, a, b))
clahe_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
clahe_rgb = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2RGB)

# Plot side-by-side
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(img_rgb)
plt.title("Original Frame")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(clahe_rgb)
plt.title("CLAHE-Enhanced Frame")
plt.axis("off")

plt.tight_layout()

save_path = output_dir / "figure4_clahe_comparison.png"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved Figure 4 to: {save_path}")