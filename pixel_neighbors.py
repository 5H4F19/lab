import cv2
import numpy as np

def get_neighbors(image_path, x, y):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    h, w = image.shape
    neighbors_4 = []
    neighbors_8 = []
    # 4-neighbors: top, bottom, left, right
    directions_4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions_4:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            neighbors_4.append(((nx, ny), image[nx, ny]))
    # 8-neighbors: includes diagonals
    directions_8 = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1),           (0, 1),
                    (1, -1),  (1, 0),  (1, 1)]
    for dx, dy in directions_8:
        nx, ny = x + dx, y + dy
        if 0 <= nx < h and 0 <= ny < w:
            neighbors_8.append(((nx, ny), image[nx, ny]))
    print(f"Pixel ({x}, {y}) value: {image[x, y]}")
    print("4-neighbors (position, value):", neighbors_4)
    print("8-neighbors (position, value):", neighbors_8)
    return neighbors_4, neighbors_8

if __name__ == "__main__":
    # Example: choose pixel at (50, 50)
    get_neighbors('test.jpg', 50, 50)
