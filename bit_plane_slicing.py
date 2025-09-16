import cv2
import matplotlib.pyplot as plt
import numpy as np

def bit_plane_slicing(image_path):
    # Load grayscale image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    
    bit_planes = []
    for i in range(7, -1, -1):  # from MSB (7) to LSB (0)
        # Extract the i-th bit using bitwise operations
        plane = (image >> i) & 1  
        # Scale to full 0–255 for visibility (otherwise only 0/1)
        plane = plane * 255
        bit_planes.append(plane)
    
    # Titles
    titles = [f"Bit {i}" for i in range(7, -1, -1)]
    
    # Plot all 8 planes
    plt.figure(figsize=(12, 6))
    for i, (plane, title) in enumerate(zip(bit_planes, titles)):
        plt.subplot(2, 4, i+1)  # 2 rows, 4 columns
        plt.imshow(plane, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    bit_plane_slicing("test.jpg")
