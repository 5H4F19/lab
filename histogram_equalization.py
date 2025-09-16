import cv2
import matplotlib.pyplot as plt


def histogram_and_equalization(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    # Histogram of original
    hist_orig = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Histogram equalization
    image_eq = cv2.equalizeHist(image)
    hist_eq = cv2.calcHist([image_eq], [0], None, [256], [0, 256])
    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(2, 2, 2)
    plt.plot(hist_orig, color='black')
    plt.title('Original Histogram')
    plt.xlim([0, 256])
    plt.subplot(2, 2, 3)
    plt.imshow(image_eq, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')
    plt.subplot(2, 2, 4)
    plt.plot(hist_eq, color='black')
    plt.title('Equalized Histogram')
    plt.xlim([0, 256])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    histogram_and_equalization('low_contrast_test.jpg')
