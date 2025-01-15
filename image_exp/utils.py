import os

from torch.utils.data import DataLoader
from torchvision.transforms import transforms

def get_transforms():
    """
    get the transforms to apply to each image in the dataset
    this will read the image, resize it to 128x128, convert it to a tensor and normalize the image
    :return: torchvision.transforms - transforms to apply to the images
    """
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(), # convert the image to a tensor with values between 0 and 1
        # normalize the image with mean and std
        # the (0.485, 0.456, 0.406) and (0.229, 0.224, 0.225) are RGB values for the mean and std respectively
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
# The DataFolderManager class is used to manage the train, test and val directories
class DataFolderManager:
    def __init__(self, images_dir):
        self.images_dir = images_dir
        if not os.path.exists(images_dir):
            raise ValueError(f"The images directory {images_dir} does not exist")
        self.train_dir = os.path.join(images_dir, 'train')
        self.test_dir = os.path.join(images_dir, 'test')
        self.val_dir = os.path.join(images_dir, 'val')

    # check if the folders exists
    def data_folders_exist(self):
        """
        check if the train, test and val folders exist
        :return: bool - True if the folders exist, False otherwise
        """
        result = os.path.exists(self.train_dir) and os.path.exists(self.test_dir) and os.path.exists(self.val_dir)
        if not result:
            print("The train, test and val folders do not exist")
        return result

    # get the path to the train, test and val directories
    def get_train_dir(self):
        return self.train_dir

    def get_test_dir(self):
        return self.test_dir

    def get_val_dir(self):
        return self.val_dir

    def get_images_dir(self):
        return self.images_dir

    def data_exists(self):
        if not self.data_folders_exist():
            return False

        if (len(os.listdir(self.train_dir)) == 0 or
                len(os.listdir(self.test_dir)) == 0 or
                len(os.listdir(self.val_dir)) == 0):
            print("The train, test or val folders are empty")
            return False
        return True

