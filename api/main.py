from fastapi import FastAPI
import joblib
import numpy as np
import tensorflow as tf
import xgboost as xgb
import os
from pathlib import Path
import uvicorn
import pandas as pd

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent  
forecast_model_path = BASE_DIR / "models" / "best_forecast_model"
cls_model_path = BASE_DIR / "models" / "best_cls_model" / "xgb_model.json"
scaler_path = BASE_DIR / "models" / "scalers" / "scaler.pkl"

# Load mô hình
forecast_model = tf.keras.models.load_model(str(forecast_model_path))
cls_model = xgb.Booster()
cls_model.load_model(str(cls_model_path))

scaler = joblib.load(scaler_path)

df = pd.read_csv(f"{BASE_DIR}/data/processed/daily_weather.csv", parse_dates=["date"])
df.set_index("date", inplace=True)

@app.get("/")
def home():
    return {"message": "Weather Forecast API is running!"}

@app.get("/historical/")
def get_historical_data():
    """Trả về dữ liệu thời tiết 7 ngày cuối"""
    return df.tail(7).to_dict(orient="index")

@app.get("/forecast/")
def get_forecast():
    """Dự báo thời tiết 7 ngày tiếp theo"""
    last_known_data = df.iloc[-56:].values  # Ensure the input has 56 timesteps
    response = []
    for i in range(7):
        prediction = forecast_model.predict(np.expand_dims(last_known_data, axis=0))
        predicted_coco = cls_model.predict(xgb.DMatrix(prediction.reshape(1, -1)))[0]

        future_date = df.index[-1] + pd.Timedelta(days=i+1)
        response.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "temp_max": prediction[0][0],
            "temp_min": prediction[0][1],
            "prcp_mean": prediction[0][2],
            "coco_mode": int(predicted_coco)
        })

        last_known_data = np.append(last_known_data[8:], prediction, axis=0)  # Update the input for the next prediction
    
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)