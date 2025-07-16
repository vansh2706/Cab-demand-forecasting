import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# Assuming preprocess and utils are in the same directory or accessible via PYTHONPATH
from preprocess import load_data, feature_engineer
from utils import save_model


def train_model(data_path, model_output_path='models/xgb_model.pkl'):
    """
    Loads historical data, preprocesses it, trains an XGBoost model,
    and saves the trained model.

    Args:
        data_path (str): Path to the historical_rides.csv file.
        model_output_path (str): Path to save the trained model.
    """
    print(f"Loading data from {data_path}...")
    df = load_data(data_path)
    if df is None or df.empty:
        print("No data loaded or data is empty. Exiting training.")
        return

    print("Performing feature engineering...")
    # Assuming 'demand' is the target column created during feature engineering
    # Or, you'd calculate it here if not part of feature_engineer directly
    # For this example, let's assume 'demand' is a column to be predicted.
    # If your demand is count based (e.g., number of rides per hour/location),
    # ensure feature_engineer aggregates the data to create this target.
    # For simplicity, let's assume the raw data already has a 'demand' column or
    # feature_engineer creates 'features' and a 'target' series.

    # A common way for demand: Aggregate rides by time/location
    # This part needs to be carefully designed based on your 'historical_rides.csv' content
    # For demonstration, let's assume feature_engineer returns a DataFrame with features
    # and we manually define a 'demand' column for the target.
    # You'll need to adapt this significantly based on your actual data and how you define demand.

    # Example: If your historical_rides.csv has 'timestamp', 'latitude', 'longitude', 'ride_id'
    # You'd typically aggregate:
    # df['timestamp_hour'] = df['timestamp'].dt.floor('H')
    # df['location_zone'] = some_geo_binning_function(df['latitude'], df['longitude'])
    # demand_df = df.groupby(['timestamp_hour', 'location_zone']).agg(demand=('ride_id', 'count')).reset_index()
    # Then feature_engineer would operate on demand_df.

    # For this placeholder, let's assume 'demand' is already a column and feature_engineer prepares other X values.
    # If your preprocess.py also returns the target, adapt this.

    # Placeholder: Let's create dummy features and target if not present
    if 'demand' not in df.columns:
        print("Warning: 'demand' column not found. Creating dummy 'demand' and 'feature_col'. Please adapt 'preprocess.py'.")
        df['demand'] = df.groupby(
            df.index // 10).transform('count').iloc[:, 0] + 1  # Dummy demand
        df['feature_col_1'] = df['demand'] * 0.5 + 2  # Dummy feature

    # This function should return features (X) and target (y)
    X, y = feature_engineer(df)

    # Placeholder if feature_engineer returns only X and y is separate or inferred
    if not isinstance(X, pd.DataFrame) or y is None:
        print("feature_engineer did not return expected X and y. Attempting default from df...")
        if 'demand' in df.columns:
            y = df['demand']
            # Drop the target and non-feature columns for X
            X = df.drop(columns=['demand'], errors='ignore')
            # If feature_engineer processed some columns, ensure X only has processed ones
            # For simplicity, let's take all numeric columns for X
            X = X.select_dtypes(include=['number'])
        else:
            raise ValueError(
                "Could not find 'demand' column or extract features. Adapt demand_model.py and preprocess.py.")

    if X.empty or y.empty:
        print("Features or target are empty after preprocessing. Cannot train model.")
        return

    print(f"Shape of features (X): {X.shape}, Shape of target (y): {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print("Training XGBoost model...")
    # Using a regressor as demand is typically a count/number
    model = xgb.XGBRegressor(objective='reg:squarederror',
                             n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    print("Evaluating model performance...")
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R-squared: {r2:.4f}")

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    save_model(model, model_output_path)
    print(f"Model saved to {model_output_path}")

    return model


if _name_ == '_main_':
    # Example usage:
    # First, make sure you have a dummy historical_rides.csv or actual data
    # Create a dummy CSV for testing if it doesn't exist
    if not os.path.exists('data/historical_rides.csv'):
        print("Creating dummy historical_rides.csv for demonstration...")
        dummy_data = {
            'timestamp': pd.to_datetime(['2025-01-01 10:00:00', '2025-01-01 11:00:00', '2025-01-01 12:00:00', '2025-01-01 13:00:00',
                                         '2025-01-02 10:00:00', '2025-01-02 11:00:00', '2025-01-02 12:00:00', '2025-01-02 13:00:00']),
            'latitude': [26.5, 26.51, 26.52, 26.53, 26.5, 26.51, 26.52, 26.53],
            'longitude': [77.0, 77.01, 77.02, 77.03, 77.0, 77.01, 77.02, 77.03],
            'ride_id': range(1, 9),
            'demand': [10, 12, 15, 11, 9, 11, 14, 10]  # Example demand values
        }
        pd.DataFrame(dummy_data).to_csv(
            'data/historical_rides.csv', index=False)
        print("Dummy data created at data/historical_rides.csv")

    os.makedirs('models', exist_ok=True)  # Ensure models directory exists
    trained_model = train_model('data/historical_rides.csv')
    if trained_model:
        print("Model training completed.")
    else:
        print("Model training failed.")
