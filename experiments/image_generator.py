from PIL import Image
import numpy as np
import cv2

from logger.logger import logger


class ImageGenerator:
    def __init__(self, color_scheme: str = 'RGB', size = (100, 100), color = (255, 255, 255)):
        self.color_scheme = color_scheme
        self.size = size
        self.color = color

    def generate_image(self) -> Image:
        return Image.new(self.color_scheme, self.size, self.color)

    def dispaly_image(self, image: Image):
        logger.info("Displaying the image...")
        # get image properties
        logger.info(f"Image size: {image.size}")
        logger.info(f"Image mode: {image.mode}")
        logger.info(f"Image format: {image.format}")
        logger.info(f"Image bands: {image.getbands()}")
        logger.info(f"Image color scheme: {self.color_scheme}")
        image.show()

    def print(self):
        logger.info(f"Color scheme: {self.color_scheme}\nSize: {self.size}\nColor: {self.color}")

class ImageGeneratorOpenCV:
    def __init__(self, size = (100, 100), color = (255, 255, 255)):
        self.size = size
        self.color = color

    def generate_image(self):
        return np.full((self.size[0], self.size[1], 3), self.color, dtype=np.uint8)

    def display_image(self, image):
        logger.info("Displaying the image...")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        logger.info("Image displayed.")

    def print(self):
        logger.info(f"Size: {self.size}\nColor: {self.color}")

if __name__ == "__main__":
    color_grey_scale = (128, 128, 128)
    image_generator_opencv = ImageGeneratorOpenCV(color=color_grey_scale)
    image = image_generator_opencv.generate_image()
    image_generator_opencv.print()
    image_generator_opencv.display_image(image)