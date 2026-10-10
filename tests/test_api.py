"""
test_api.py
Comprehensive test suite for the Smart Energy Consumption Prediction FastAPI backend.
Uses FastAPI TestClient to test all endpoints, input validations, error handling,
anomaly detection, and recommendation logic.
"""

import sys
import os
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ensure macOS ARM compatibility is applied
import src.compat

import unittest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app
from backend.services.model_service import ModelService

class TestEnergyPredictionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.svc = ModelService.get_instance()
        assert cls.svc.is_loaded, f"Model artifacts should load: {cls.svc.load_error}"

    def test_01_health_check(self):
        """Test GET /health returns 200 with healthy status and model_loaded=True."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])
        self.assertIn("GRU", data["model_type"])

    def test_02_model_info(self):
        """Test GET /model-info returns architectural specs, features, and benchmark metrics."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("GRU", data["model_type"])
        self.assertEqual(data["input_window"], "24 hours")
        self.assertEqual(data["prediction_horizon"], "1 hour (Next Hour)")
        self.assertEqual(len(data["features_used"]), 19)
        self.assertEqual(data["total_features"], 19)
        
        # Verify verified test metrics
        metrics = data["metrics"]
        self.assertAlmostEqual(metrics["MAE_Wh"], 406.86, places=1)
        self.assertAlmostEqual(metrics["RMSE_Wh"], 607.26, places=1)
        self.assertAlmostEqual(metrics["R2_score"], 0.6489, places=2)
        
        # Verify baseline comparison entries (5 models)
        baselines = data["baseline_comparison"]
        self.assertEqual(len(baselines), 5)
        names = [b["model"] for b in baselines]
        self.assertIn("Previous Hour (Persistence)", names)
        self.assertIn("LSTM (Baseline)", names)
        self.assertIn("GRU (Best Selected)", names)

    def test_03_predict_next_hour_default(self):
        """Test POST /predict with default reference sequence."""
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("predicted_consumption", data)
        self.assertIn("unit", data)
        self.assertEqual(data["unit"], "kWh")
        self.assertIn("level", data)
        self.assertIn("timestamp", data)
        
        # From predict.py output, next-hour prediction for the latest sequence is ~2.16 kWh (2157.4 Wh)
        self.assertGreater(data["predicted_consumption"], 0.0)
        self.assertAlmostEqual(data["predicted_consumption"], 2.16, delta=0.2)

    def test_04_predict_next_hour_custom_valid(self):
        """Test POST /predict with a custom valid 24-hour sequence."""
        sample = self.svc.get_latest_sample(24)
        records = sample.reset_index().rename(columns={"index": "timestamp"}).to_dict(orient="records")
        # Format timestamps as string
        for r in records:
            r["timestamp"] = str(r["timestamp"])
            
        response = self.client.post("/predict", json={"data": records})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertAlmostEqual(data["predicted_consumption"], 2.16, delta=0.2)

    def test_05_predict_invalid_short_sequence(self):
        """Test POST /predict with fewer than 24 rows returns 422 Unprocessable Entity."""
        short_data = [{"timestamp": f"2007-11-28 1{i}:00:00", "energy_consumption": 1000.0} for i in range(5)]
        response = self.client.post("/predict", json={"data": short_data})
        self.assertEqual(response.status_code, 422)
        self.assertIn("At least 24 hourly time steps required", response.json()["detail"])

    def test_06_predict_next_24_hours(self):
        """Test POST /predict-next-24-hours returns exactly 24 recursive forecast steps."""
        response = self.client.post("/predict-next-24-hours", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        predictions = data["predictions"]
        self.assertEqual(len(predictions), 24)
        
        # Verify sequence indexing
        for i, p in enumerate(predictions, start=1):
            self.assertEqual(p["step"], i)
            self.assertGreaterEqual(p["predicted_consumption"], 0.0)
            self.assertEqual(p["unit"], "kWh")
            self.assertIn("hour_label", p)
            self.assertIn("level", p)
            
        self.assertGreater(data["total_predicted_kwh"], 0.0)
        self.assertIn("peak_hour", data)
        self.assertIn("lowest_hour", data)

    def test_07_analytics_default_and_historical(self):
        """Test POST /analytics and GET /analytics/historical."""
        # 1. POST /analytics
        resp = self.client.post("/analytics", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["total_consumption"], 0.0)
        self.assertGreater(data["average_consumption"], 0.0)
        self.assertEqual(data["unit"], "kWh")
        self.assertIn(":", data["peak_hour"])
        
        # 2. GET /analytics/historical
        resp_hist = self.client.get("/analytics/historical")
        self.assertEqual(resp_hist.status_code, 200)
        data_hist = resp_hist.json()
        self.assertEqual(len(data_hist["actual_vs_predicted_24h"]), 24)
        self.assertEqual(len(data_hist["actual_vs_predicted_7d"]), 168)
        self.assertEqual(len(data_hist["diurnal_pattern"]), 24)
        self.assertEqual(len(data_hist["day_of_week_pattern"]), 7)

    def test_08_anomaly_detection(self):
        """Test POST /anomaly-detection with normal and extreme anomalous inputs."""
        # Test default
        resp = self.client.post("/anomaly-detection", json={})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_evaluated"], 24)
        self.assertIn("results", data)
        
        # Test normal consumption sequence (~1.0 kWh during day)
        normal_data = [
            {"timestamp": f"2007-11-28 {h:02d}:00:00", "actual_consumption": 1.1}
            for h in range(12, 16)
        ]
        resp_norm = self.client.post("/anomaly-detection", json={"data": normal_data})
        self.assertEqual(resp_norm.status_code, 200)
        for item in resp_norm.json()["results"]:
            self.assertFalse(item["is_anomaly"])
            self.assertIn("Normal", item["anomaly_status"])
            
        # Test extreme anomaly (9.5 kWh at 3 AM when typical baseline is ~0.35 kWh)
        surge_data = [
            {"timestamp": "2007-11-28 03:00:00", "actual_consumption": 9.5}
        ]
        resp_surge = self.client.post("/anomaly-detection", json={"data": surge_data})
        self.assertEqual(resp_surge.status_code, 200)
        surge_res = resp_surge.json()
        self.assertEqual(surge_res["anomalies_detected"], 1)
        self.assertTrue(surge_res["results"][0]["is_anomaly"])
        self.assertGreater(surge_res["results"][0]["z_score"], 2.5)

    def test_09_recommendations(self):
        """Test POST /recommendations returns data-driven energy saving strategies."""
        resp = self.client.post("/recommendations", json={})
        self.assertEqual(resp.status_code, 200)
        recs = resp.json()["recommendations"]
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn("category", r)
            self.assertIn("priority", r)
            self.assertIn("title", r)
            self.assertIn("recommendation", r)

    def test_10_missing_model_handling(self):
        """Test graceful degradation when model is unavailable."""
        orig_model = self.svc.model
        orig_is_loaded = self.svc.is_loaded
        try:
            # Simulate unloaded model
            self.svc.model = None
            self.svc.is_loaded = False
            self.svc.load_error = "Simulated missing model file"
            
            # GET /health should report model_loaded: False
            resp_health = self.client.get("/health")
            self.assertEqual(resp_health.status_code, 200)
            self.assertFalse(resp_health.json()["model_loaded"])
            
            # POST /predict should return 503 Service Unavailable
            resp_pred = self.client.post("/predict", json={})
            self.assertEqual(resp_pred.status_code, 503)
            self.assertIn("Model service is not loaded", resp_pred.json()["detail"])
        finally:
            # Restore service state
            self.svc.model = orig_model
            self.svc.is_loaded = orig_is_loaded
            self.svc.load_error = None

if __name__ == "__main__":
    unittest.main()
