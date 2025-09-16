import cv2
import numpy as np
import matplotlib.pyplot as plt

def laplacian_sharpening(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    # Apply Laplacian filter
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    # Sharpened image: original + Laplacian
    sharpened = cv2.addWeighted(image, 1, laplacian_abs, 1, 0)
    # Display
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(sharpened, cmap='gray')
    plt.title('Sharpened (Laplacian)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    laplacian_sharpening('test.jpg')
