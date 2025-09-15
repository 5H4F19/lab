import matplotlib.pyplot as plt
import cv2
import numpy as np


def quantize_gray_levels(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    levels = [2, 4, 8, 16, 256]
    quantized_images = []
    for l in levels:
        step = 256 // l
        quantized = (image // step) * step
        quantized_images.append(quantized)
    titles = [f"{l} levels" for l in levels]
    plt.figure(figsize=(15, 3))
    for i, (img, title) in enumerate(zip(quantized_images, titles)):
        plt.subplot(1, len(levels), i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    quantize_gray_levels('test.jpg')
