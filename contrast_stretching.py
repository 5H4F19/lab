import cv2
import numpy as np
import matplotlib.pyplot as plt

def contrast_stretching(image_path):
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Invalid image path.")
        return
    
    # Get min and max
    r_min, r_max = np.min(image), np.max(image)
    
    # Apply contrast stretching formula
    stretched = ((image - r_min) / (r_max - r_min) * 255).astype(np.uint8)

    # Plot results
    plt.figure(figsize=(12, 6))

    # Original
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.hist(image.ravel(), bins=256, range=[0,256], color='black')
    plt.title("Original Histogram")

    # Stretched
    plt.subplot(2, 2, 3)
    plt.imshow(stretched, cmap='gray')
    plt.title("Contrast Stretched Image")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.hist(stretched.ravel(), bins=256, range=[0,256], color='black')
    plt.title("Stretched Histogram")

    plt.tight_layout()
    plt.show()


# Example usage
contrast_stretching("test2.png")
