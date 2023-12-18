import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

def predict_datacenter_id(requested_array, model_name="models/lstm87.keras"):
    """
    Predicts the data center ID for a given set of task parameters using a saved Keras LSTM model.

    Args:
        requested_array: A list containing the following task parameters:
            - TaskID: integer
            - TaskFileSize: integer
            - TaskOutputFileSize: integer
            - TaskFileLength: integer
        model_name: The path to the saved Keras LSTM model file (default: "models/lstm87.keras").

    Returns:
        The predicted data center ID (integer).

    Raises:
        ValueError: If the requested array length is not equal to the expected number of features.
    """
  
    expected_feature_count = 4

    if len(requested_array) != expected_feature_count:
        raise ValueError(f"Expected {expected_feature_count} features, but received {len(requested_array)}.")

    # Build the data dictionary and reshape the input data
    data_dict = {
        "TaskID": requested_array[0],
        "TaskFileSize": requested_array[1],
        "TaskOutputFileSize": requested_array[2],
        "TaskFileLength": requested_array[3],
    }
    input_data = np.array([[data_dict["TaskID"], data_dict["TaskFileSize"], data_dict["TaskOutputFileSize"], data_dict["TaskFileLength"]]])
    input_data_lstm = input_data.reshape(input_data.shape[0], 1, input_data.shape[1])
    loaded_model = load_model(model_name)
    # Make predictions using the loaded model
    predicted_probabilities = loaded_model.predict(input_data_lstm)
    predicted_class = np.argmax(predicted_probabilities, axis=1)
    predicted_datacenter_id = int(predicted_class[0])
    return predicted_datacenter_id

    


# Example usage
requested_array = [0.1, 0, 0, 55]
predicted_datacenter_id = predict_datacenter_id(requested_array)


print(f"Predicted DataCenterID: {predicted_datacenter_id}")
