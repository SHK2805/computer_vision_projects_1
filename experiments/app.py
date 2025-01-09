from image_generator import ImageGenerator, ImageGeneratorOpenCV
from experiments.image_reader import ImageReaderOpenCV
from logger.logger import logger
from PIL import Image
import random

def generate_image():
    color_scheme: str = 'RGB'
    size = (100, 100)
    # random number between 0 and 255
    color = (random.randint(0,255),
             random.randint(0,255),
             random.randint(0,255))
    logger.info("Generating an image...")
    image_generator: ImageGenerator = ImageGenerator(color_scheme, size, color)
    image: Image = image_generator.generate_image()
    image_generator.print()
    logger.info("Image generated.")
    logger.info("Displaying the image...")
    image_generator.dispaly_image(image)

def generate_image_opencv(size = (500, 500), color = (0, 100, 255)):

    # in BGR format for OpenCV
    logger.info("Generating an image...")
    image_generator_opencv = ImageGeneratorOpenCV(size, color)
    image = image_generator_opencv.generate_image()
    image_generator_opencv.print()
    logger.info("Image generated.")
    logger.info("Displaying the image...")
    image_generator_opencv.display_image(image)

def read_image():
    image_path = "../images/test_image.png"
    logger.info("Reading an image...")
    image_reader = ImageReaderOpenCV(image_path)
    image_reader.print()
    logger.info("Image read.")
    logger.info("Displaying the image...")
    image_reader.show()

if __name__ == "__main__":
    color_grey_scale = (128, 128, 128)
    # generate_image_opencv(color=color_grey_scale)
    read_image()
