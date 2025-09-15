import matplotlib.pyplot as plt
import cv2


def downsample_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    factors = [2, 4, 8]
    subsampled_images = [image[::f, ::f] for f in factors]
    titles = ['Original', 'Downsampled x2', 'Downsampled x4', 'Downsampled x8']
    images = [image] + subsampled_images
    plt.figure(figsize=(12, 4))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    downsample_grayscale('test.jpg')
