# Deploying Machine Learning Models with Streamlit and Docker

## Introduction

In the world of Machine Learning Engineering, deploying models into production is just as crucial as developing them. A well-trained model is only valuable if it can be reliably and efficiently served to end-users or integrated into existing systems. This project demonstrates the complete lifecycle of deploying a machine learning model, from training and inference testing to building a Dockerized application for seamless distribution and scalability.

Docker, a containerization platform, is a cornerstone of modern ML Engineering. It allows developers to package applications and their dependencies into portable containers that run consistently across different environments. With Docker, ML models can be deployed with ease, ensuring reproducibility and compatibility, whether on a developer’s local machine or in the cloud.

---

## Project Steps

This project comprises the following key steps:

### 1. Training and Saving the Model
- Train a machine learning model using a dataset.
- Save the trained model to disk in a format that can be easily loaded for inference (e.g., `.pkl` or `.joblib`).

### 2. Testing Model Inference
- Perform single prediction testing to validate the model’s functionality.
- Implement batch prediction testing to evaluate the model’s performance on multiple inputs.

### 3. Building the Streamlit Application
- Develop a user-friendly web interface using Streamlit.
- The app should allow users to interact with the model for predictions, supporting both single and batch inputs.

### 4. Writing the Dockerfile and Building the Docker Image
- Create a `Dockerfile` to define the container’s environment and dependencies.
- Build a Docker image containing the Streamlit app and all necessary files.

### 5. Running the Container and Testing the Application
- Use the Docker image to run a container.
- Test the Streamlit app inside the container to ensure it functions as expected.

### 6. Stopping and Cleaning Docker Resources
- Stop the running container.
- Remove the container and Docker image to clean up the environment.

---

## How to Run the Project

Follow these steps to execute the project:

### Prerequisites
1. Install Python (>= 3.8).
2. Install Docker (https://www.docker.com/).
3. Clone this repository:
```bash
   git clone https://github.com/Baldezo313/ML-Docker-Streamlit.git
   ```
4. (Optional but recommended) Create a virtual environment:
   ```b   ash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
5. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
6. Download and Install Docker Desktop

### 1. Train the Model
Run the script to train the model and save it:
```bash
python3 scripts/train_model.py
```

### 2. Test Inference
Run the inference script for single and batch predictions:
```bash
python3 scripts/test_inference.py
```

### 3. Build the Streamlit App
Start the Streamlit app locally:
```bash
streamlit run app/app.py
```

### 4. Build the Docker Image
Create the Docker image from the `Dockerfile`:
```bash
docker build -t ml-model-app .
```

### 5. Run the Docker Container
Start the container using the built image:
```bash
docker run -p 8501:8501 ml-model-app
```
Access the Streamlit app in your browser at `http://localhost:8501`.

### 6. Stop and Clean Docker Resources
1. Stop the running container:
   ```bash
   docker ps
   docker stop <container_id>
   ```
2. Remove the container:
   ```bash
   docker rm <container_id>
   ```
3. Remove the Docker image:
   ```bash
   docker rmi ml-model-app
   ```

---

## Key Features
- **Reproducibility:** The project uses Docker to ensure consistent behavior across different environments.
- **Modular Design:** Separation of model training, inference, and deployment logic for clarity and maintainability.
- **User-Friendly Interface:** The Streamlit app provides a simple and intuitive way for users to interact with the model.

---
