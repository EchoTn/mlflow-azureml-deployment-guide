"""
Script to deploy a machine learning model to Azure ML as a managed online endpoint.

This script automates the following tasks:
- Connects to an Azure Machine Learning workspace using the default Azure credential.
- Retrieves a registered ML model and environment from the workspace.
- Creates (or updates) a managed online endpoint with AAD token authentication and disabled public access.
- Deploys the model to the endpoint with specified configuration including:
    - Instance type and count.
    - Code path and scoring script.
    - Monitoring data collectors for request, response, model inputs, and outputs.
- Routes 100% traffic to the deployed model version.

Key components:
- `connect_to_ml_workspace`: Authenticates and connects to Azure ML workspace.
- `get_registered_model`: Retrieves the latest registered model by name.
- `get_environment`: Retrieves the latest registered environment by name.
- `create_online_endpoint`: Creates or updates the online endpoint with defined security settings.
- `deploy_model`: Deploys the model to the endpoint with the associated environment and scoring logic.
- `main`: Orchestrates the full deployment flow.

Prerequisites:
- Valid Azure credentials.
- Model and environment already registered in Azure ML workspace.
- `score.py` script present in the root directory.
"""

import logging
from azure.ai.ml import MLClient
from azure.ai.ml.entities import (
    DataAsset,
    DataCollector,
    ManagedOnlineDeployment,
    ManagedOnlineEndpoint,
    DeploymentCollection,
)
from azure.identity import DefaultAzureCredential
from azureml.exceptions import UserErrorException


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define data collections
collections = {
    "request": DeploymentCollection(
        enabled=True,
        data=DataAsset(name="input_requests"),
    ),
    "response": DeploymentCollection(
        enabled=True,
        data=DataAsset(name="endpt_response"),
    ),
    "model_inputs": DeploymentCollection(
        enabled=True,
        data=DataAsset(name="model_inputs"),
    ),
    "model_outputs": DeploymentCollection(
        enabled=True,
        data=DataAsset(name="model_outputs"),
    ),
}


def connect_to_ml_workspace():
    """
    Connect to Azure ML Workspace.
    """
    try:
        ml_client = MLClient.from_config(credential=DefaultAzureCredential())
        logger.info("Successfully connected to Azure ML Workspace.")
        return ml_client
    except UserErrorException as e:
        logger.error(f"Failed to connect to workspace: {e}")
        raise


def get_registered_model(ml_client, model_name):
    """
    Retrieve the registered model from Azure ML.
    """
    try:
        model = ml_client.models.get(model_name, label="latest")
        logger.info(f"Model {model_name} retrieved successfully.")
        return model
    except Exception as e:
        logger.error(f"Failed to retrieve model '{model_name}': {e}")
        raise


def get_environment(ml_client, env_name):
    """
    Fetch the environment using the provided configuration parameters:
    """
    try:
        # Get environment from the Azure ML workspace
        environment = ml_client.environments.get(name=env_name, label="latest")

        logger.info(f"Environment {env_name} fetched successfully.")
        return environment

    except Exception as e:
        logger.error(f"Failed to fetch environment '{env_name}': {e}")
        raise


def create_online_endpoint(ml_client, endpoint_name):
    """
    Create or update an online endpoint in Azure ML.
    """
    endpoint = ManagedOnlineEndpoint(
        name=endpoint_name,
        description="Iris flower classification model endpoint",
        auth_mode="aad_token",  # Using Azure Active Directory token for authentication
        public_network_access="disabled",  # Restrict access to private network only
    )

    try:
        ml_client.online_endpoints.begin_create_or_update(endpoint).wait()
        logger.info(f"Endpoint '{endpoint_name}' created or updated successfully.")
    except Exception as e:
        logger.error(f"Failed to create or update endpoint '{endpoint_name}': {e}")
        raise
    return endpoint


def deploy_model(ml_client, model, endpoint_name, deployment_name, environment):
    """
    Deploy the model to the online endpoint.
    """
    deployment = ManagedOnlineDeployment(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=model,  # The model to be deployed
        environment=environment,  # Specify the registered environment
        code_path=".",  # Path to the folder containing your code and scoring script
        scoring_script="score.py",  # Path to your custom scoring script
        instance_type="Standard_DS2_v2",  # Specify compute for deployment
        instance_count=1,  # Number of instances to deploy
        data_collector=DataCollector(
            collections=collections
        ),  # Attach a DataCollector for logging requests/responses
        app_insights_enabled=True,  # Enable Application Insights for telemetry and diagnostics
    )

    try:
        ml_client.online_deployments.begin_create_or_update(deployment).wait()
        logger.info(
            f"Model deployed successfully to endpoint: {endpoint_name}/{deployment_name}"
        )
    except Exception as e:
        logger.error(f"Failed to deploy the model: {e}")
        raise


def main():
    """
    Main function to execute the deployment process.
    """
    # Connect to Azure ML workspace
    ml_client = connect_to_ml_workspace()

    # Model retrieval
    model_name = "iris-model"  # Replace with your model's name
    model = get_registered_model(ml_client, model_name)

    # Environemnt retrieval
    env_name= "iris-env"
    environment = get_environment(ml_client, env_name)

    # Configure online endpoint
    endpoint_name = "iris-model-endpoint"
    endpoint = create_online_endpoint(ml_client, endpoint_name)

    # Deploy the model
    deployment_name = "iris-deployment"
    deploy_model(ml_client, model, endpoint_name, deployment_name, environment)

    # Update the traffic
    endpoint.traffic = {deployment_name: 100}
    ml_client.online_endpoints.begin_create_or_update(endpoint).wait()


if __name__ == "__main__":
    main()
