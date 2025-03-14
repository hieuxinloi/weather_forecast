import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
from pathlib import Path 

BASE_DIR = Path(__file__).resolve().parent.parent  

# Load scaler and models
scaler_path = BASE_DIR / "models/scalers/scaler.pkl"
lstm_model_path = BASE_DIR / "models/best_forecast_model"
xgb_model_path = BASE_DIR / "models/best_cls_model/xgboost_model.json"

scaler = joblib.load(scaler_path)
lstm_model = tf.keras.models.load_model(lstm_model_path)
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(str(xgb_model_path))

# features
features = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres']

# Load weather data
data_path = BASE_DIR / "data/processed/weatherQN_2021_2025_processed.csv"
data = pd.read_csv(data_path, parse_dates=["time"])
data.set_index("time", inplace=True)

# predicted results path
results_path = BASE_DIR / "data/results/results.csv"

# forecast next 7 days
def forecast_next_7_days():
    last_input = data[-56:][features].values.reshape(1, 56, len(features))
    predictions = []

    for _ in range(7):  
        pred = lstm_model.predict(last_input, verbose=0)
        predictions.append(pred)
        last_input = np.concatenate([last_input[:, 8:, :], pred], axis=1)

    predictions = np.concatenate(predictions, axis=1)
    predictions_original = scaler.inverse_transform(predictions.reshape(-1, len(features))).reshape(56, len(features))

    future_dates = pd.date_range(data.index[-1] + pd.Timedelta(hours=3), periods=56, freq="3h")
    df_predictions = pd.DataFrame(predictions_original, columns=features, index=future_dates)
    return df_predictions

# classify coco
def classify_coco(dataframe):
    if 'coco' in dataframe.columns:
        dataframe = dataframe.drop(columns=['coco'])

    X_cls = dataframe.copy()
    X_cls['hour'] = X_cls.index.hour
    X_cls['day'] = X_cls.index.day
    X_cls['month'] = X_cls.index.month
    X_cls['year'] = X_cls.index.year
    X_cls['dayofweek'] = X_cls.index.dayofweek

    expected_features = ['temp', 'dwpt', 'rhum', 'prcp', 'wdir', 'wspd', 'pres',
                         'hour', 'dayofweek', 'day', 'month', 'year']
    
    X_cls = X_cls[expected_features]
    coco_preds = xgb_model.predict(X_cls)
    dataframe['coco'] = coco_preds
    return dataframe

# summarize daily data
def summarize_daily_data(df):
    df = df.copy()

    if 'coco' not in df.columns:
        df['coco'] = np.nan  

    daily_summary = df.resample('D').agg({
        'temp': ['max', 'min'],
        'coco': lambda x: x.mode()[0] if not x.mode().empty else np.nan,
        'rhum': 'mean', 
        'wspd': 'mean'   
    })

    daily_summary.columns = ['temp_max', 'temp_min', 'coco_mode', 'rhum', 'wspd']
    daily_summary = daily_summary.reset_index()

    daily_summary['temp_max'] = daily_summary['temp_max'].round().astype(int)
    daily_summary['temp_min'] = daily_summary['temp_min'].round().astype(int)
    daily_summary['rhum'] = daily_summary['rhum'].round().astype(int) 
    daily_summary['wspd'] = daily_summary['wspd'].round().astype(int)  

    return daily_summary

# save results
def save_forecast_results():
    df_forecast = forecast_next_7_days()
    df_forecast = classify_coco(df_forecast)
    df_forecast = df_forecast.reset_index().rename(columns={'index': 'time'}) # reset index to time
    df_forecast.to_csv(results_path, index=False)

# get display data
def get_display_data():
    last_7_days = summarize_daily_data(data[-56:])

    save_forecast_results()

    full_forecast = pd.read_csv(results_path, parse_dates=["time"])
    full_forecast.set_index("time", inplace=True)

    next_7_days = summarize_daily_data(full_forecast)

    last_7_days['date'] = last_7_days['time'].dt.strftime("%a, %d/%m")
    next_7_days['date'] = next_7_days['time'].dt.strftime("%a, %d/%m")

    return last_7_days.to_dict(orient="records"), next_7_days.to_dict(orient="records")

