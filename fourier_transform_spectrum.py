import cv2
import numpy as np
import matplotlib.pyplot as plt

def fourier_transform_spectrum(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    # Compute 2D Fourier Transform
    f = np.fft.fft2(image)
    print(f)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    # Display
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum (log scale)')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    fourier_transform_spectrum('test.jpg')
