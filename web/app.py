from flask import Flask, render_template
from utils import get_display_data

app = Flask(__name__)

COCO_MAPPING = {
    0: "clear",
    1: "fair",
    2: "cloudy",
    3: "fog",
    4: "light_rain",
    5: "rain",
    6: "heavy_rain",
    7: "rain_shower",
    8: "heavy_rain_shower"
}

@app.route("/")
def index():
    last_7_days, next_7_days = get_display_data()

    for day in next_7_days:
        day["coco_mode_name"] = COCO_MAPPING.get(day["coco_mode"], "unknown")

    return render_template("index.html", last_7_days=last_7_days, next_7_days=next_7_days)

if __name__ == "__main__":
    app.run(debug=True)