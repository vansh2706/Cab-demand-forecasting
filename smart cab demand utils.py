import joblib
import os


def save_model(model, file_path):
    """
    Saves a trained model to a specified file path using joblib.

    Args:
        model: The trained machine learning model.
        file_path (str): The path where the model should be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(model, file_path)
        print(f"Model successfully saved to {file_path}")
    except Exception as e:
        print(f"Error saving model to {file_path}: {e}")


def load_model(file_path):
    """
    Loads a trained model from a specified file path using joblib.

    Args:
        file_path (str): The path to the saved model file.

    Returns:
        The loaded machine learning model.
    """
    try:
        model = joblib.load(file_path)
        print(f"Model successfully loaded from {file_path}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading model from {file_path}: {e}")
        return None

# You can add other utility functions here, e.g., logging setup


def setup_logging():
    """
    Sets up basic logging for the application.
    """
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    print("Logging setup complete.")


if _name_ == '_main_':
    # Example usage of utils
    # Create a dummy model for testing save/load
    from sklearn.linear_model import LinearRegression
    dummy_model = LinearRegression()
    test_model_path = 'test_models/dummy_model.pkl'

    save_model(dummy_model, test_model_path)
    loaded_model = load_model(test_model_path)

    if loaded_model:
        print("Dummy model saved and loaded successfully.")
    else:
        print("Failed to save or load dummy model.")

    # Clean up test files
    if os.path.exists(test_model_path):
        os.remove(test_model_path)
        print(f"Cleaned up {test_model_path}")
    if os.path.exists(os.path.dirname(test_model_path)):
        os.rmdir(os.path.dirname(test_model_path))
        print(f"Cleaned up {os.path.dirname(test_model_path)}")
