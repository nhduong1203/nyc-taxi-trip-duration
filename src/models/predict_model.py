import pickle
import numpy as np  # If you'll be using numpy arrays for test data


class PredictModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self.loaded_model = None

    def load_model(self):
        with open(self.model_path, 'rb') as file:
            self.loaded_model = pickle.load(file)

    def predict(self, test_data):
        if self.loaded_model is None:
            raise Exception("Model not loaded. Call load_model method first.")

        # Assuming test_data is a numpy array or pandas DataFrame
        predictions = self.loaded_model.predict(test_data)
        return predictions

