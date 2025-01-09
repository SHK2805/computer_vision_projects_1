import cv2

from logger.logger import logger


class ImageReaderOpenCV:
    def __init__(self, image_path: str = "../images/test_image.png"):
        self.image_path = image_path
        self.image = self.read()

    def read(self):
        image = cv2.imread(self.image_path)
        logger.info(f"Image read from {self.image_path}")
        logger.info(f"Image shape: {image.shape}")
        logger.info(f"Image size: {image.size}")
        return image

    def show(self):
        cv2.imshow("Image", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def print(self):
        logger.info(f"Image path: {self.image_path}")

if __name__ == "__main__":
    image_reader = ImageReaderOpenCV()
    image_reader.print()
    image_reader.show()