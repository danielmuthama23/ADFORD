import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Define CIFAR-10 class labels
LABELS = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Load fine-tuned model
def load_model():
    # Load a ResNet18 model without pre-trained weights
    model = models.resnet18(weights=None)  # Use `weights=None` instead of `pretrained=False`
    
    # Adjust the final fully connected layer for 10 classes (CIFAR-10)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    
    # Load the fine-tuned weights
    model.load_state_dict(
        torch.load("model/fine_tuned.pth", map_location=torch.device("cpu"), weights_only=True)
    )
    
    # Set the model to evaluation mode
    model.eval()
    return model

# Define image transformations
def transform_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match ResNet input size
        transforms.ToTensor(),          # Convert PIL image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Make predictions
def predict(model, image):
    # Transform the image and add batch dimension
    image = transform_image(image)
    
    # Perform inference
    with torch.no_grad():  # Disable gradient computation for inference
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)  # Get the predicted class index
    
    # Return the predicted class label
    return LABELS[predicted.item()]