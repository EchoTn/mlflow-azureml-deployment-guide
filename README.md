# 🚀 Serving MLflow Models on Azure ML  
*A Complete Guide to Real-Time Inference with Custom Scoring & Monitoring*

This repository supports the Medium article: 
📖 [Serving MLflow Models on Azure ML](https://medium.com/henkel-data-and-analytics/serving-mlflow-models-on-azure-ml-deploy-with-online-endpoints-and-custom-scoring-scripts-f69d40bdcb55)

> "**Deploying machine learning models shouldn’t mean wrestling with infrastructure.**"
> With Azure ML's Managed Online Endpoints, you get scalable, secure, real-time inferencing without managing VMs or Kubernetes.

## 🧭 What You'll Learn

This guide walks you through:

- ✅ Training & logging a model with **MLflow**
- 🧪 Creating a custom **Azure ML Environment**
- ✍️ Writing a custom **scoring script** with input/output monitoring
- 🚀 Deploying to **Azure ML Managed Online Endpoints**
- 📡 Sending authenticated prediction requests via REST

## 📁 Project Structure

```
├── aml/
│   ├── environment/
│   │   ├── conda.yaml      # Conda environment definition
│   │   └── create_env.py   # Registers the Conda environment
│   └── online-endpoint/        
│   │   ├── deploy.yaml     # Script to deploy model to endpoint
│   │   └── score.py        # Scoring script with monitoring integration
│
├── src/
│   └── train.py            # MLflow-based training and model logging
│
├── .gitignore              # Files/directories to ignore in Git
├── README.md               # Project overview and instructions
└── requirements.txt        # Python dependencies for local development
```

## ⚙️ Prerequisites

- Azure subscription & ML workspace
- Python 3.8+
- Azure ML SDK v2 installed in your environment: pip install azure-ai-ml
- mlflow-skinny
- azureml-mlflow (Mlflow extension on Azure)
- scikit-learn
- pandas
 
## 🚀 Quickstart

### 1️⃣ Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/mlflow-azureml-deployment-guide.git
cd mlflow-azureml-deployment-guide
```

### 2️⃣ Train & Log Model
Train a simple classifier on the Iris dataset and log it to Azure ML.

```bash
python src/train.py
```
This will:
- Train a logistic regression model
- Log metrics and model to your Azure ML workspace
- Register the model under the name "iris-model"

### 3️⃣ Create a custom Azure ML Environment
```bash
python aml/environment/create_env.py
```
This script:
- Loads the conda.yaml file
- Registers the environment as "iris-env" in your Azure ML workspace

### 3️⃣ Deploy to Online Endpoint

Deploy your model with:
```bash
python aml/online-enpoint/deploy.py
```
This script:
- Retrieves the registered environment and model
- Deploys it with the custom scoring script score.py
- Creates a Managed Online Endpoint in Azure ML

### 4️⃣ Send Authenticated Requests
You can call your endpoint using the following python code sample:

```python
import requests

token = "<your_access_token>"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

payload = {
    "input_data": [{
        "sepal length (cm)": 5.1,
        "sepal width (cm)": 3.5,
        "petal length (cm)": 1.4,
        "petal width (cm)": 0.2
    }]
}

url = f"https://<region>.inference.ml.azure.com/endpoint/iris-model-endpoint/score"
response = requests.post(url, headers=headers, json=payload)
print(response.json())
```
If the request is successful you will get a 200 response status and the predicted flower species.

❗If you encounter a `401: JWT is missing` error, make sure you’ve correctly fetched your bearer token and included it in the request headers.