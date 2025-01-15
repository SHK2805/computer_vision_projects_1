import os

from PIL import Image  # PIL is the Python Imaging Library
from torch.utils.data import Dataset, DataLoader

from utils import DataFolderManager, get_transforms


class ImageDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        # location of the images
        self.images_dir = images_dir
        self.transform = transform
        # store the image paths
        self.image_paths = []
        # store the image labels
        self.labels = []

        # get the image paths and labels
        # the images are stored in the following way
        # images_dir
        #   - train
        #       - class_1
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_2
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_n
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #   - test
        #       - class_1
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_2
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_n
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #   - val
        #       - class_1
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_2
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        #       - class_n
        #           - image_1.jpg
        #           - image_2.jpg
        #           - ...
        # the labels are the indices of the class directories i.e. the path to each image file with its name is the label
        # enumerate the top images directory
        # os.listdir returns a list of the files in the directory
        # class_dir is the name of the class directory here like class_1, class_2, ...
        # label stores the index of the class directory like 0, 1, 2, ... for each directory
            # like class_1 is 0, class_2 is 1, class_3 is 2, ...
        # we pass in the path to the train, test and val images directories and get the list of the class directories in each and the index of each class directory
        for label, class_dir in enumerate(os.listdir(images_dir)):
            # get the path to the class directory
            class_path = os.path.join(images_dir, class_dir)
            # enumerate the class directory and get the path to each image file
            for image_path in os.listdir(class_path):
                self.image_paths.append(os.path.join(class_path, image_path))
                # store the label
                self.labels.append(label)

    def __len__(self):
        """
        return the number of images in the dataset
        :return: int - number of images in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        get the image and its label at the given index
        :param idx: int - index of the image
        :return: torch.Tensor - image, int - label
        """
        # read the image
        image = Image.open(self.image_paths[idx]).convert('RGB') # convert to RGB to ensure 3 channels
        # get the label
        label = self.labels[idx]
        # apply the transform if it exists
        if self.transform:
            # apply the transform to the image
            # the torch vision accepts only PIL images or tensors or batches of tensors
            # if we are doing any open cv operations then we need to convert that image to a PIL image
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return self.labels

    def get_image_paths(self):
        return self.image_paths

def get_image_loader(images_dir, transform, batch_size, shuffle=True):
    """
    get the DataLoader for the images
    :param images_dir: str - path to the images directory
    :param transform: torchvision.transforms - transform to apply to the images
    :param batch_size: int - batch size
    :param shuffle: bool - shuffle the images or not
    :return: DataLoader - DataLoader for the images
    """
    # create the dataset
    img_dataset = ImageDataset(images_dir, transform)
    # create the DataLoader
    return DataLoader(img_dataset, batch_size=batch_size, shuffle=shuffle)


if __name__ == '__main__':
    folder_manager = DataFolderManager('../data/images')
    if not folder_manager.data_exists():
        raise ValueError("Data does not exist")
    train_data_loader = get_image_loader(folder_manager.get_train_dir(), transform=get_transforms(), batch_size=32, shuffle=True)
    test_data_loader = get_image_loader(folder_manager.get_test_dir(), transform=get_transforms(), batch_size=32, shuffle=False)
    val_data_loader = get_image_loader(folder_manager.get_val_dir(), transform=get_transforms(), batch_size=32, shuffle=False)
    for img, labels in val_data_loader:
        print(img.shape, labels)
        break



