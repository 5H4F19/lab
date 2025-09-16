import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_high_pass_filter(image_path, radius=30):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    # Fourier Transform
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    # Create Ideal High-Pass Filter mask
    mask = np.ones((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius * radius
    mask[mask_area] = 0
    # Apply mask and inverse DFT
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    # Display
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(img_back, cmap='gray')
    plt.title(f'IHPF Filtered (radius={radius})')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ideal_high_pass_filter('test.jpg', radius=30)
