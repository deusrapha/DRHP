import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

def apply_clahe(image):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L (Lightness) channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)
    
    # Merge the CLAHE enhanced L channel with the original a and b channels
    merged = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced_img = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return enhanced_img

def hsv_color_correction(image):
    """Demonstrates HSV color space conversion and manipulation."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Example: slightly increase saturation to make plant colors pop
    s = cv2.add(s, 20)
    
    corrected_hsv = cv2.merge((h, s, v))
    corrected_img = cv2.cvtColor(corrected_hsv, cv2.COLOR_HSV2BGR)
    return corrected_img

def extract_lbp_features(image):
    """Extracts Local Binary Patterns (LBP) for texture representation."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # LBP parameters
    radius = 3
    n_points = 8 * radius
    
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    return lbp

def extract_glcm_features(image):
    """Extracts Gray-Level Co-occurrence Matrix (GLCM) features."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # GLCM parameters
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    
    glcm = graycomatrix(gray, distances=distances, angles=angles, symmetric=True, normed=True)
    
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    
    return {
        'contrast': contrast,
        'dissimilarity': dissimilarity,
        'homogeneity': homogeneity,
        'energy': energy,
        'correlation': correlation
    }

def process_and_visualize(image_path):
    """Reads an image, applies techniques, and visualizes the results."""
    if not os.path.exists(image_path):
        print(f"Image not found at {image_path}")
        return
        
    original = cv2.imread(image_path)
    if original is None:
        print("Failed to load image.")
        return
        
    # Apply Enhancements
    clahe_img = apply_clahe(original)
    hsv_img = hsv_color_correction(original)
    
    # Extract Features
    lbp_feat = extract_lbp_features(original)
    glcm_feat = extract_glcm_features(original)
    
    print("GLCM Features (Sample):")
    for k, v in glcm_feat.items():
        print(f"  {k}: {v:.4f}")

    # Visualization
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    
    # OpenCV uses BGR, Matplotlib uses RGB
    axs[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    axs[0, 1].set_title('CLAHE Enhanced (Contrast)')
    axs[0, 1].axis('off')
    
    axs[1, 0].imshow(cv2.cvtColor(hsv_img, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title('HSV Color Corrected')
    axs[1, 0].axis('off')
    
    # LBP is grayscale
    axs[1, 1].imshow(lbp_feat, cmap='gray')
    axs[1, 1].set_title('LBP Texture Features')
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('enhancement_results.png')
    print("Saved visualization to enhancement_results.png")
    # plt.show() # Uncomment to view interactively

if __name__ == "__main__":
    # Test with one of the extracted frames
    test_image_path = "labbled/labbled/clip (100).jpg"
    process_and_visualize(test_image_path)
