import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

class Resnet50PretrainedModel:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(self.image_path).convert('RGB')
        # Load the pretrained model
        self.model = models.resnet50(weights='ResNet50_Weights.DEFAULT')
        # Set the model to evaluation mode
        self.model.eval()
        # Define the image transformation
        self.transform = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            ])


    def predict(self):
        # Apply the transformations to the input image and convert it to a tensor
        transform_image = self.transform( self.image).unsqueeze(0)
        # Perform the forward pass
        with torch.no_grad():
            model_prediction = self.model(transform_image)
            # Get the class index of the predicted class
            predicted_class = torch.argmax(model_prediction).item()
        return predicted_class

if __name__ == '__main__':
    input_image_path = '../data/images/light_house.jpeg'
    prediction = Resnet50PretrainedModel(input_image_path)
    # print the prediction
    # The output will be the class number of the predicted class
    # 436 is for car - beach wagon, station wagon, wagon, estate car, beach waggon, station wagg, waggon',
    # 437 is the class for 'lighthouse'
    print(f"Predicted class: {prediction.predict()}")