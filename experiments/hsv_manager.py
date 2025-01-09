import os
import cv2
import numpy as np

class HSVManager:
    def __init__(self, image: str = "../images/test_image.png"):
        if not image:
            raise ValueError("Image not found.")

        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file {image} not found.")

        self.image = cv2.imread(image)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.hsv_values = {
            'H': [0, 180],
            'S': [0, 255],
            'V': [0, 255]
        }

    # get image
    def get_image(self):
        return self.image

    # get warm_cool_image_prediction image
    def get_hsv_image(self):
        return self.hsv_image

    # display image and warm_cool_image_prediction image in the same window
    def display_images(self):
        images = np.hstack((self.get_image(), self.get_hsv_image()))
        cv2.imshow("Images", images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    hsv_manager = HSVManager()
    hsv_manager.display_images()



