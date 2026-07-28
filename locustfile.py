"""
locustfile.py
Maize Leaf Disease Classifier — load testing script for the API.

Simulates a flood of prediction requests against the API. Run with:

    locust -f locustfile.py --host http://localhost:8000

or headless (used for the report's results):

    locust -f locustfile.py --host http://localhost:8000 \
        --users 20 --spawn-rate 5 --run-time 60s --headless \
        --csv results/1worker
"""

import pathlib
import random

from locust import HttpUser, task, between

SAMPLE_DIR = pathlib.Path(__file__).parent / "data" / "test"
SAMPLE_IMAGES = list(SAMPLE_DIR.glob("*/*.jpg")) if SAMPLE_DIR.exists() else []


class MaizeScanUser(HttpUser):
    """Simulates a field agent using the app: mostly predicting, occasionally checking status."""

    wait_time = between(0.5, 2.0)

    @task(8)
    def predict(self):
        if not SAMPLE_IMAGES:
            return
        image_path = random.choice(SAMPLE_IMAGES)
        with open(image_path, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("leaf.jpg", f, "image/jpeg")},
                name="/predict",
            )

    @task(2)
    def health(self):
        self.client.get("/health", name="/health")

    @task(1)
    def model_info(self):
        self.client.get("/model-info", name="/model-info")
