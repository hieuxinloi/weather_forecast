# Weather Forecast Web App

The **Weather Forecast** project is a web application for weather prediction using a **GRU + XGBoost** model. It is built with **Flask** to display 7-day weather forecasts.  

## Main Features  

- **Weather forecasting** based on historical data using **GRU + XGBoost**.  
- **User-friendly web interface** displaying key weather information:  
  - Daily highest temperature  
  - Daily lowest temperature  
  - Average wind speed
  - Average precipitation  
  - Weather condition code (`coco`)  
- **Flask-based backend** serving API for the web interface.  
- **Integration of TensorFlow and XGBoost** for prediction.  

## Project Structure  

```bash
weather_forecast/
│── data/                        # Weather data (raw & processed & results)
│── images/                      # Images to write readme.md
│── models/                      # Forecast & Classification models
│── notebooks/                   # Notebooks for data processing & training
│── web/                         # Flask web app code
│   ├── static/                  # CSS, icons
│   ├── templates/               # HTML templates
│   ├── app.py                   # Flask server
│   ├── utils.py                 # Helper functions
│── requirements.txt             # Required dependencies
│── README.md                    # Project documentation
```

## Installation
### 1. Clone the repository
```bash
git clone https://github.com/thangthewinner/weather_forecast.git
cd weather_forecast
```

### 2. Set up a virtual environment & install dependencies
```bash
python -m venv venv
venv\Scripts\activate  
pip install -r requirements.txt
```


### 3. Run the Flask application
```bash
cd web
python app.py
```
Open your browser and visit: http://127.0.0.1:5000/

## Web Preview

![UI Preview](/images/ui_preview.png)