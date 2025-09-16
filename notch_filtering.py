import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_periodic_noise(image, amplitude=40, freq=8):
    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)
    noise = amplitude * np.sin(2 * np.pi * freq * X / cols)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image

def notch_filter(shape, notch_centers, radius=5):
    rows, cols = shape
    mask = np.ones((rows, cols), np.uint8)
    for center in notch_centers:
        y, x = center
        Y, X = np.ogrid[:rows, :cols]
        dist = (X - x) ** 2 + (Y - y) ** 2 <= radius ** 2
        mask[dist] = 0
    return mask

def notch_filtering(image_path, amplitude=40, freq=8, radius=5):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    # Add synthetic periodic noise
    noisy = add_periodic_noise(image, amplitude, freq)
    # Fourier Transform
    f = np.fft.fft2(noisy)
    fshift = np.fft.fftshift(f)
    rows, cols = noisy.shape
    crow, ccol = rows // 2, cols // 2
    # Notch filter centers (positive and negative frequency)
    notch_centers = [
        (crow, ccol + freq),
        (crow, ccol - freq)
    ]
    mask = notch_filter((rows, cols), notch_centers, radius)
    # Apply mask and inverse DFT
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    # Display
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(noisy, cmap='gray')
    plt.title('Noisy (Periodic)')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(img_back, cmap='gray')
    plt.title('Notch Filtered')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    notch_filtering('test.jpg', amplitude=40, freq=8, radius=5)
