
# GNN Molecular Property Prediction

## Project Overview
This project develops a **Graph Neural Network (GNN)** using **PyTorch Geometric** to predict **molecular properties** from molecular graphs. The model is trained on the **QM9 dataset**, which contains molecular structures and corresponding properties derived from **Density Functional Theory (DFT)** calculations. The goal is to leverage **deep learning and graph-based techniques** to accelerate molecular property prediction, reducing the need for expensive quantum chemistry simulations.

## Purpose
- Predict molecular properties from graph representations of molecules.
- Accelerate material and drug discovery by reducing dependency on computationally expensive simulations.
- Enable scalable and distributed training on **AWS GPU clusters** for improved performance.

---

## Features
**Graph Neural Networks (GNNs)** for molecular property prediction  
Uses **PyTorch Geometric** for graph-based deep learning  
Trained on **QM9 dataset** (molecular structures and DFT-calculated properties)  
**Distributed training on AWS** for scalable model training  
**REST API with Flask** for real-time predictions  
**Dockerized Deployment** for easy scalability  

---

## Technologies Used
| Technology  | Description |
|-------------|------------|
| **Python** | Programming language for model development |
| **PyTorch Geometric** | Deep learning framework for graph neural networks |
| **PyTorch** | Base deep learning framework |
| **RDKit** | Chemical informatics library for handling molecular data |
| **AWS EC2** | Cloud computing for distributed training |
| **Flask** | API framework for deploying predictions |
| **Docker** | Containerization for scalable deployment |



---

## How to Run the Project

### 1️⃣ **Install Dependencies**
pip install -r deployment/requirements.txt
2️⃣ Train the Model
python src/train.py
3️⃣ Run Distributed Training on AWS
python src/distributed_train.py
4️⃣ Start the API Server
python deployment/app.py
The API will be available at: http://localhost:5000/predict

5️⃣ Docker Deployment (Optional)
docker build -t gnn-api .
docker run -p 5000:5000 gnn-api

---

## Results & Performance
The GNN model successfully predicts molecular properties with high accuracy.
Using AWS GPU clusters significantly speeds up training through distributed computation.
Deployment via Flask API and Docker ensures easy access and scalability.

## Future Improvements
🔹 Implement Graph Attention Networks (GATs) for better performance
🔹 Train on larger molecular datasets beyond QM9
🔹 Deploy via AWS Lambda and API Gateway for serverless inference
🔹 Optimize training with hyperparameter tuning



