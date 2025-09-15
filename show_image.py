import matplotlib.pyplot as plt
import cv2

def show_image_matplotlib(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"File '{image_path}' does not exist or is not a valid image.")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Image')
    plt.show()

if __name__ == "__main__":
    show_image_matplotlib('test.jpg')
