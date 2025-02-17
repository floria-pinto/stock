import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the LSTM model (make sure the .h5 file is in the same directory)
model = load_model('lstm_model.h5')

# Function to predict stock prices using the loaded LSTM model
def predict_stock(input_data):
    # Ensure the input_data is reshaped if necessary for the model
    # Example: Reshaping input to 3D for LSTM if needed
    input_data = np.reshape(input_data, (input_data.shape[0], 1, 1))  # Adjust based on model requirements

    # Make predictions
    predictions = model.predict(input_data)
    return predictions.flatten()  # Flatten to a 1D array if needed for plotting
