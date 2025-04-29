"""
Create and register a custom Azure ML environment.

This script performs the following tasks:
1. Connects to an Azure ML workspace using the `DefaultAzureCredential` and local config.
2. Defines a custom environment using:
    - A Conda environment YAML file (`conda.yaml`)
    - A base Docker image for inference
3. Registers or updates the environment in the Azure ML workspace.

Environment Details:
- Name: `iris-env`
- Docker base image: Azure ML's official MLflow inference image
- Conda file: Should be located at the root of the project and named `conda.yaml`

Usage:
Run this script as a standalone module to create or update the custom environment
needed for training or inference.

Dependencies:
- azure-ai-ml
- azure-identity
- conda.yaml (required in the working directory)

Example:
$ python create_environment.py
"""

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential


def main():
    # Connect to the Azure ML workspace
    ml_client = MLClient.from_config(
        DefaultAzureCredential(),
    )

    # Define the environment settings
    env_definition = Environment(
        name="iris-env",
        conda_file="conda.yaml",
        image="mcr.microsoft.com/azureml/mlflow-ubuntu20.04-py38-cpu-inference:latest",
    )

    # Create or update the environment in Azure ML
    print("Creating or updating environment...")
    env_creation_job = ml_client.environments.create_or_update(env_definition)
    print(
        f"Environment {env_creation_job.name} version {env_creation_job.version} registered successfully."
    )


if __name__ == "__main__":
    main()
