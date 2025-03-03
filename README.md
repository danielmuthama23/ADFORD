## **AI Developer Assignment - Image Classification API**

### **📌 Overview**
This project implements an AI-powered image classification API using **FastAPI** and **PyTorch**. The application fine-tunes a pre-trained deep learning model and serves predictions via a REST API.

### **📂 Project Structure**
\`\`\`
📂 ai_developer_assignment/
│── 📂 app/
│   ├── __init__.py            # Package initializer
│   ├── main.py                # FastAPI application
│   ├── model.py               # Model loading and prediction logic
│   ├── requirements.txt       # Python dependencies
│── Dockerfile                 # Docker setup for deployment
│── README.md                  # Project documentation
│── venv/                      # Virtual environment (optional)
\`\`\`

### **🚀 Features**
- Loads and fine-tunes a pre-trained model (e.g., ResNet, EfficientNet).
- Preprocesses and transforms images before inference.
- Serves predictions via a FastAPI-based REST API.
- Supports deployment using Docker.

---

###
**⚙️ Installation**
### **1️⃣ Clone the Repository**
\`\`\`bash
git clone https://github.com/danielmuthama23/ADFORD.git
cd ADFORD
\`\`\`

### **2️⃣ Create a Virtual Environment (Optional)**
\`\`\`bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
\`\`\`

### **3️⃣ Install Dependencies**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

---

## **🚀 Running the API**
### **1️⃣ Start the FastAPI Server**
\`\`\`bash
uvicorn app.main:app --reload
\`\`\`
The API will be available at:  
🔗 **http://127.0.0.1:8000**

---

## **🖼️ Making Predictions**
You can send an image to the API using **cURL**:

\`\`\`bash
curl -X 'POST' 'http://127.0.0.1:8000/predict/' -F 'file=@sample.jpg'
\`\`\`
### **📌 Expected Response**
\`\`\`json
{
  "filename": "sample.jpg",
  "prediction": "cat"
}
\`\`\`

---

## **🐳 Docker Deployment**
### **1️⃣ Build the Docker Image**
\`\`\`bash
docker build -t ai-classifier .
\`\`\`
### **2️⃣ Run the Container**
\`\`\`bash
docker run -p 8000:8000 ai-classifier
\`\`\`

---

## **📌 Technologies Used**
- **FastAPI** – API framework
- **PyTorch** – Deep Learning framework
- **Torchvision** – Model and image transformations
- **PIL (Pillow)** – Image processing
- **Docker** – Containerization

---

## **💡 Future Improvements**
- Add GPU acceleration using **CUDA**
- Deploy to **AWS Lambda / Hugging Face Spaces**
- Implement **batch processing for inference**

---

## **📄 License**
This project is **MIT Licensed**.

---


