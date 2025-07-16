import pandas as pd
import numpy as np


def load_data(file_path):
    """
    Loads historical ride data from a CSV file.

    Args:
        file_path (str): The path to the historical_rides.csv file.

    Returns:
        pd.DataFrame: Loaded DataFrame, or None if file not found.
    """
    try:
        df = pd.read_csv(file_path)
        # Ensure timestamp is datetime object
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def clean_data(df):
    """
    Performs basic data cleaning steps.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    if df is None:
        return None
    # Drop rows with any missing values
    df = df.dropna()
    # Remove duplicates
    df = df.drop_duplicates()
    # Basic outlier handling (example: remove very unrealistic coordinates)
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
        df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
    return df


def feature_engineer(df):
    """
    Creates new features from the raw data suitable for demand prediction.
    This function should also define the target variable (demand).

    Args:
        df (pd.DataFrame): The input DataFrame from load_data.

    Returns:
        tuple: (pd.DataFrame: Features (X), pd.Series: Target (y))
    """
    df = clean_data(df.copy())
    if df is None or df.empty:
        print("DataFrame is empty after cleaning, cannot perform feature engineering.")
        # Return empty DataFrame and Series if no data
        return pd.DataFrame(), pd.Series(dtype=float)

    # Ensure timestamp column exists and is datetime type
    if 'timestamp' not in df.columns or not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        raise ValueError(
            "DataFrame must contain a 'timestamp' column of datetime type.")

    # Time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['quarter'] = df['timestamp'].dt.quarter
    df['is_weekend'] = ((df['day_of_week'] == 5) | (
        df['day_of_week'] == 6)).astype(int)
    df['is_peak_hour'] = ((df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9) |
                          (df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19)).astype(int)

    # Location-based features (simple binning for demonstration)
    # In a real scenario, you'd use geo-hashing (geohash), grid systems, or pre-defined zones.
    # For simplicity, let's round lat/long to create coarse "zones"
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df['lat_zone'] = (df['latitude'] * 100).astype(int)
        df['lon_zone'] = (df['longitude'] * 100).astype(int)

        # Combine into a single categorical zone ID
        df['location_zone'] = df['lat_zone'].astype(
            str) + '_' + df['lon_zone'].astype(str)
    else:
        print("Warning: Latitude/longitude columns not found for zone creation.")
        df['location_zone'] = 'unknown_zone'  # Placeholder

    # Aggregate demand:
    # If your raw data is per-ride, you need to define what "demand" means.
    # Common definitions: rides per hour per location.
    # For training, you'd group by time and location, then count rides.

    # Assuming 'ride_id' exists and 'demand' is not directly in the raw data
    # If 'demand' column is already in the raw data (e.g., pre-aggregated), skip this.
    if 'ride_id' in df.columns:
        print("Aggregating rides to create demand target...")
        # Group by all time and location features to get demand for each unique context
        # Convert necessary columns to string/category before grouping if they are numeric IDs
        features_for_groupby = ['timestamp', 'hour_of_day', 'day_of_week', 'month',
                                'day_of_year', 'week_of_year', 'quarter', 'is_weekend', 'is_peak_hour',
                                'location_zone']

        # Ensure 'timestamp' is used to define the aggregation interval (e.g., hourly)
        df['time_slot'] = df['timestamp'].dt.floor('H')  # Aggregate by hour

        # Demand calculation: count of rides per time slot and location zone
        demand_df = df.groupby(['time_slot', 'location_zone']).agg(
            demand=('ride_id', 'count'),  # Target variable
            # Average latitude for the zone/slot
            latitude=('latitude', 'mean'),
            # Average longitude for the zone/slot
            longitude=('longitude', 'mean'),
            # Take mode or first value for other features
            hour_of_day=('hour_of_day', 'first'),
            day_of_week=('day_of_week', 'first'),
            month=('month', 'first'),
            day_of_year=('day_of_year', 'first'),
            week_of_year=('week_of_year', 'first'),
            quarter=('quarter', 'first'),
            is_weekend=('is_weekend', 'first'),
            is_peak_hour=('is_peak_hour', 'first')
        ).reset_index()

        # Drop original timestamp, as we have time_slot now
        demand_df = demand_df.drop(columns=['time_slot'])

        df_processed = demand_df
    else:
        # If 'ride_id' is not present, assume 'demand' column already exists in raw data
        # and other features are already engineered or will be selected.
        print("No 'ride_id' found, assuming 'demand' column exists and is target.")
        df_processed = df.copy()

    # One-hot encode categorical features (e.g., day of week, month, location zone)
    # Be careful with high-cardinality features like location_zone; they can explode features.
    # For location_zone, you might consider embedding or simpler approaches for large numbers of zones.
    # Add 'location_zone' if it doesn't create too many columns
    categorical_cols = ['day_of_week', 'month', 'quarter']

    # If location_zone has too many unique values, treat it differently or use only for grouping
    # Threshold for OHE
    if 'location_zone' in df_processed.columns and df_processed['location_zone'].nunique() < 1000:
        categorical_cols.append('location_zone')
    else:
        print("Warning: 'location_zone' has high cardinality, not one-hot encoded for this example. Consider other encoding methods.")
        if 'location_zone' in df_processed.columns:
            # Convert to category type for better memory usage, but don't OHE for this example
            df_processed['location_zone'] = df_processed['location_zone'].astype(
                'category').cat.codes  # Label encode

    df_processed = pd.get_dummies(
        df_processed, columns=categorical_cols, drop_first=True)

    # Define features (X) and target (y)
    if 'demand' in df_processed.columns:
        y = df_processed['demand']
        X = df_processed.drop(columns=['demand'], errors='ignore')
    else:
        raise ValueError(
            "Target 'demand' column not found after feature engineering. Check data and logic.")

    # Remove non-numeric or irrelevant columns that might remain
    X = X.select_dtypes(include=np.number)

    # Store columns used during training for consistent prediction later
    # This is a critical step for preprocess_for_prediction
    global _training_features
    _training_features = list(X.columns)
    print(f"Features used in training: {_training_features}")

    return X, y


# This global variable will store the columns that the model was trained on.
# It's crucial for ensuring the input to the deployed model has the same columns in the same order.
_training_features = None


def preprocess_for_prediction(input_df):
    """
    Preprocesses a single or multiple new input rows for prediction.
    Must apply the exact same transformations as feature_engineer().

    Args:
        input_df (pd.DataFrame): DataFrame with raw input data (e.g., from API request).
                                 Must contain 'timestamp', 'latitude', 'longitude'.

    Returns:
        pd.DataFrame: Processed features ready for model prediction.
    """
    if _training_features is None:
        raise RuntimeError(
            "Model's training features not set. Ensure feature_engineer was run at least once during model training.")

    df_pred = input_df.copy()

    if 'timestamp' not in df_pred.columns or not pd.api.types.is_datetime64_any_dtype(df_pred['timestamp']):
        df_pred['timestamp'] = pd.to_datetime(df_pred['timestamp'])

    # Apply the same time-based feature engineering
    df_pred['hour_of_day'] = df_pred['timestamp'].dt.hour
    df_pred['day_of_week'] = df_pred['timestamp'].dt.dayofweek
    df_pred['month'] = df_pred['timestamp'].dt.month
    df_pred['day_of_year'] = df_pred['timestamp'].dt.dayofyear
    df_pred['week_of_year'] = df_pred['timestamp'].dt.isocalendar().week.astype(int)
    df_pred['quarter'] = df_pred['timestamp'].dt.quarter
    df_pred['is_weekend'] = ((df_pred['day_of_week'] == 5) | (
        df_pred['day_of_week'] == 6)).astype(int)
    df_pred['is_peak_hour'] = ((df_pred['hour_of_day'] >= 7) & (df_pred['hour_of_day'] <= 9) |
                               (df_pred['hour_of_day'] >= 17) & (df_pred['hour_of_day'] <= 19)).astype(int)

    # Apply the same location-based feature engineering
    if 'latitude' in df_pred.columns and 'longitude' in df_pred.columns:
        df_pred['lat_zone'] = (df_pred['latitude'] * 100).astype(int)
        df_pred['lon_zone'] = (df_pred['longitude'] * 100).astype(int)
        df_pred['location_zone'] = df_pred['lat_zone'].astype(
            str) + '_' + df_pred['lon_zone'].astype(str)
    else:
        # Match training if lat/lon not present
        df_pred['location_zone'] = 'unknown_zone'

    # One-hot encode categorical features, ensuring consistent columns
    categorical_cols = ['day_of_week', 'month', 'quarter']
    # Re-add location_zone if it was used for OHE during training
    if any('location_zone_' in col for col in _training_features):
        categorical_cols.append('location_zone')

    df_processed_pred = pd.get_dummies(
        df_pred, columns=categorical_cols, drop_first=True)

    # Ensure all columns from training are present, fill missing with 0 (for OHE columns not present)
    # and drop extra columns
    final_features = pd.DataFrame(columns=_training_features)
    final_features = pd.concat(
        [final_features, df_processed_pred], ignore_index=True)
    final_features = final_features.reindex(
        columns=_training_features, fill_value=0)

    # Drop original timestamp and any other non-feature columns
    final_features = final_features.drop(columns=[
                                         'timestamp', 'latitude', 'longitude', 'lat_zone', 'lon_zone'], errors='ignore')

    # For location_zone that was label encoded, ensure it exists and is int type
    if 'location_zone' in _training_features and 'location_zone' in df_processed_pred.columns:
        if not pd.api.types.is_integer_dtype(final_features['location_zone']):
            # If it was label encoded, convert incoming categorical value to its code
            # This requires mapping. For simplicity, if not OHE, it's safer to treat as a raw feature
            # For this example, if it was label encoded (cat.codes), this part is tricky without a stored mapping.
            # If it was used directly as location_zone in training and not OHE, then it should be present.
            pass  # Already handled by reindex and fill_value=0 for dummy columns

    return final_features.select_dtypes(include=np.number)
