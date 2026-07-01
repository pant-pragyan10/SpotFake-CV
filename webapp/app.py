"""Tiny live demo: camera + upload UI backed by the same ScreenRecaptureDetector as predict.py.

Run:
    python webapp/app.py
Then open http://localhost:5050
"""

from __future__ import annotations

import io
import logging
import sys
import time
from pathlib import Path

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from src.models.inference import get_detector

app = Flask(__name__)
get_detector()  # warm the model at import time, not on the first request


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    file = request.files.get("image")
    if file is None:
        return jsonify(error="No image uploaded"), 400

    try:
        image = Image.open(io.BytesIO(file.read()))
        image.load()
    except UnidentifiedImageError:
        return jsonify(error="Could not read that file as an image"), 400

    start = time.perf_counter()
    try:
        probability = get_detector().predict_proba_image(image)
    except Exception as exc:
        return jsonify(error=str(exc)), 500
    latency_ms = (time.perf_counter() - start) * 1000.0

    return jsonify(
        probability=probability,
        label="screen" if probability >= 0.5 else "real",
        latency_ms=round(latency_ms, 1),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
