"""
Scoring script for an Azure ML managed online endpoint with monitoring.

This script defines the logic for initializing and running inference on a deployed ML model.
It includes input/output monitoring using Azure AI Monitoring SDK and exposes a REST API
interface for real-time predictions.

Key Functions:
- `init()`: Called once when the endpoint starts. It loads the trained model, sets up
  input/output data collectors, and initializes a correlation context for traceability.
- `_load_model()`: Loads a serialized model artifact from the Azure ML-mounted path.
- `run(raw_data)`: Called on each prediction request. It:
    - Parses and transforms incoming JSON input into a DataFrame.
    - Logs and collects input features.
    - Runs the prediction using the loaded model.
    - Logs and collects prediction results.
    - Returns predictions or error details as an AMLResponse.

Expected Input (JSON):
{
    "input_data": [
        {
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2
        }
    ]
}

Expected Output:
A list of predicted class labels (e.g., `[0]`).

Dependencies:
- Azure ML SDK
- pandas
- joblib
- scikit-learn
- azureml-ai-monitoring

Notes:
- Uses AAD-based secured endpoint access.
- Assumes the model was trained with a pandas DataFrame with matching feature names.
- Requires `score.py` to be packaged with the model during deployment.
"""

from azureml.ai.monitoring import Collector
from azureml.ai.monitoring.context import BasicCorrelationContext
from azureml.contrib.services.aml_request import rawhttp
from azureml.contrib.services.aml_response import AMLResponse
import logging
import joblib
import os
import pandas as pd
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.propagate = True


def init():
    """
    Function which is run once within the endpoint right after it is created.

    Initializes the following:
    - model
    - collectors for input and output monitoring
    """
    global model, inputs_collector, outputs_collector

    # Initialize collectors for input and output data
    inputs_collector = Collector(
        name="model_inputs", on_error=lambda e: logging.info(f"ex:{e}")
    )
    outputs_collector = Collector(
        name="model_outputs", on_error=lambda e: logging.info(f"ex:{e}")
    )

    # Load the model
    model = _load_model()

    logger.info("Init completed")


def _load_model():
    """
    Load the pre-trained model from the path provided by Azure ML.

    Returns:
        model: The loaded model.
    """
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model/model.pkl")
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


@rawhttp
def run(raw_data):
    """
    Function which is called for every API request to make predictions.

    Args:
        request: The HTTP request containing the input data.

    Returns:
        AMLResponse: The prediction results or error response.
    """

    try:
        logging.info("Request received")
        # Decode raw HTTP request body and parse the JSON
        raw_data = raw_data.get_data().decode("utf-8")
        data = json.loads(raw_data)["input_data"]

        # Convert input data to pandas DataFrame
        input_df = pd.DataFrame(data)

        # Create a unique correlation ID for this request to enable traceability
        artificial_context = BasicCorrelationContext(id=str(uuid.uuid4()))

        # Collect input data for monitoring
        context = inputs_collector.collect(input_df, artificial_context)

        # Make the prediction using the model
        result = model.predict(input_df)

        # Collect output data for monitoring
        outputs_collector.collect(result, context)

        # Log the prediction response
        logging.info(f"Prediction response: {result}")
        return result.tolist()

    except Exception as error:
        # Log and return the error if something goes wrong
        logging.error(f"Error during prediction: {repr(error)}")
        return AMLResponse(repr(error), status_code=400)
