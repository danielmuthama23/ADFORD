# from fastapi import FastAPI, UploadFile, File
# import torch
# import torchvision.transforms as transforms
# from PIL import Image
# import io
# from app.model import load_model, predict

# app = FastAPI()

# # Load model
# model = load_model()

# @app.post("/predict/")
# async def classify_image(file: UploadFile = File(...)):
#     image = Image.open(io.BytesIO(await file.read()))
#     prediction = predict(model, image)
#     return {"prediction": prediction}

from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
from torchvision import transforms
from app.model import load_model, predict  # Ensure correct import

app = FastAPI()

# Load the model when the server starts
model = load_model()

# Define preprocessing transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Transform image
        image = transform(image).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            output = model(image)
            prediction = torch.argmax(output, dim=1).item()

        return {"filename": file.filename, "prediction": prediction}

    except Exception as e:
        return {"error": str(e)}
