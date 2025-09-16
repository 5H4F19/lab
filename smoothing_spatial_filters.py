import cv2
import numpy as np
import matplotlib.pyplot as plt

def smoothing_spatial_filters(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    # Apply averaging filters
    kernels = [(3, 3), (5, 5), (10, 10)]
    smoothed = [cv2.blur(image, k) for k in kernels]

    titles = ['Original'] + [f'kernel {k}' for k in kernels]
    images = [image] + smoothed

    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    smoothing_spatial_filters('test.jpg')
