import sys
from flask import Flask, request, jsonify, render_template
from src.pipeline.prediction_pipeline import PREDICTION_PIPELINE, CUSTOM_DATASET
from src.exception import CustomException
from src.logger import logging
import pandas as pd

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form

        input_data = CUSTOM_DATASET(
            abnormal_url=int(data["abnormal_url"]),
            phish_long_path=int(data["phish_long_path"]),
            phish_adv_number_count=int(data["phish_adv_number_count"]),
            https=int(data["https"]),
            web_ssl_valid=int(data["web_ssl_valid"]),
            suspicious_extension=int(data["suspicious_extension"]),
            phish_adv_many_subdomains=int(data["phish_adv_many_subdomains"]),
            phish_adv_long_domain=int(data["phish_adv_long_domain"]),
            phish_urgency_words=int(data["phish_urgency_words"]),
            web_security_score=int(data["web_security_score"]),
            url_len=int(data["url_len"]),
            percent_count=int(data["percent_count"]),
            plus_count=int(data["plus_count"]),
            phish_adv_hyphen_count=int(data["phish_adv_hyphen_count"]),
            phish_multiple_subdomains=int(data["phish_multiple_subdomains"]),
            phish_adv_many_params=int(data["phish_adv_many_params"]),
            question_mark_count=int(data["question_mark_count"]),
            web_xframe=int(data["web_xframe"]),
            equal_count=int(data["equal_count"]),
            phish_adv_exact_brand_match=int(data["phish_adv_exact_brand_match"])
        )

        df = pd.DataFrame([input_data.dataset()])

        pipeline = PREDICTION_PIPELINE()
        pred = pipeline.predict(df)[0]

        label_map = {
            0: "benign",
            1: "defacement",
            2: "phishing",
            3: "malware"
        }

        result = label_map[int(pred)]

        return render_template("home.html", prediction=result)

    except Exception as e:
        logging.error(str(e))
        raise CustomException(e, sys)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()

        input_data = CUSTOM_DATASET(
            abnormal_url=int(data.get("abnormal_url",0)),
            phish_long_path=int(data.get("phish_long_path",0)),
            phish_adv_number_count=int(data.get("phish_adv_number_count",0)),
            https=int(data.get("https",0)),
            web_ssl_valid=int(data.get("web_ssl_valid",0)),
            suspicious_extension=int(data.get("suspicious_extension",0)),
            phish_adv_many_subdomains=int(data.get("phish_adv_many_subdomains",0)),
            phish_adv_long_domain=int(data.get("phish_adv_long_domain",0)),
            phish_urgency_words=int(data.get("phish_urgency_words",0)),
            web_security_score=int(data.get("web_security_score",0)),
            url_len=int(data.get("url_len",0)),
            percent_count=int(data.get("percent_count",0)),
            plus_count=int(data.get("plus_count",0)),
            phish_adv_hyphen_count=int(data.get("phish_adv_hyphen_count",0)),
            phish_multiple_subdomains=int(data.get("phish_multiple_subdomains",0)),
            phish_adv_many_params=int(data.get("phish_adv_many_params",0)),
            question_mark_count=int(data.get("question_mark_count",0)),
            web_xframe=int(data.get("web_xframe",0)),
            equal_count=int(data.get("equal_count",0)),
            phish_adv_exact_brand_match=int(data.get("phish_adv_exact_brand_match",0))
        )

        df = pd.DataFrame([input_data.dataset()])
        pipeline = PREDICTION_PIPELINE()
        pred = pipeline.predict(df)[0]

        label_map = {
            0: "benign",
            1: "defacement",
            2: "phishing",
            3: "malware"
        }

        return jsonify({"prediction": label_map[int(pred)]})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
