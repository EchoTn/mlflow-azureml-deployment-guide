# 🚀 Serving MLflow Models on Azure ML  
*A Complete Guide to Real-Time Inference with Custom Scoring & Monitoring*

This repository supports the Medium article: 
📖 [Serving MLflow Models on Azure ML](https://medium.com/...)

> "**Deploying machine learning models shouldn’t mean wrestling with infrastructure.**"
> With Azure ML's Managed Online Endpoints, you get scalable, secure, real-time inferencing without managing VMs or Kubernetes.

## 🧭 What You'll Learn

This guide walks you through:

- ✅ Training & logging a model with **MLflow**
- 🧪 Creating a custom **Azure ML Environment**
- ✍️ Writing a custom **scoring script** with input/output monitoring
- 🔐 Securing inference with **OAuth2**
- 🚀 Deploying to **Azure ML Managed Online Endpoints**
- 📡 Sending authenticated prediction requests via REST

## 📁 Project Structure

```
├── aml/
│   ├── environment/
│   │   ├── conda.yaml      # Conda environment definition
│   │   └── create_env.py   # Registers the Conda environment
│   └── online-enpoint/        
│   │   ├── deploy.yaml     # Script to deploy model to endpoint
│   │   └── score.py        # Scoring script with monitoring integration
├── notebooks/
│   └── tutorial.ipynb      # Optional: interactive walkthrough
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
- Azure ML SDK v2:  
`pip install azure-ai-ml`

 
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

### 4️⃣ Send Authenticated Requests (TBA)



## 📚 Resources

[Azure ML SDK v2](https://learn.microsoft.com/azure/machine-learning/)

[MLflow Docs](https://mlflow.org/docs/latest/index.html)

[Managed Online Endpoints](https://learn.microsoft.com/azure/machine-learning/concept-endpoints)
