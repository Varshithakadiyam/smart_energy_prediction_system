"""
test_user_forecast.py
Comprehensive test suite for User-Specific Energy Forecasting from CSV Uploads.
Tests data validation, gap handling, univariate GRU forecasting, anomaly detection,
privacy session management, and API regressions.
"""

import sys
import io
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Apply macOS ARM compatibility bridge
import src.compat

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from backend.main import app

def generate_sample_user_csv(num_hours=48, start_date="2026-09-01 00:00:00", base_val=1.2, add_gaps=False):
    dates = pd.date_range(start=start_date, periods=num_hours, freq="1h")
    # Diurnal pattern
    kwh = [round(base_val + 0.6 * np.sin(2 * np.pi * d.hour / 24.0) + 0.1 * np.random.rand(), 3) for d in dates]
    df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"), "energy_consumption_kwh": kwh})
    if add_gaps:
        # Drop 2 non-contiguous rows (allowed small gaps)
        df = df.drop(index=[10, 25]).reset_index(drop=True)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    return csv_buf.getvalue().encode("utf-8")

class TestUserForecastAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_download_template(self):
        """Test GET /user-forecast/template returns downloadable CSV template."""
        response = self.client.get("/user-forecast/template")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers.get("content-type", ""))
        content = response.text
        self.assertIn("timestamp,energy_consumption_kwh", content)
        self.assertIn("2026-09-01", content)

    def test_02_upload_valid_csv(self):
        """Test POST /user-forecast/upload with valid 48-hour CSV."""
        csv_bytes = generate_sample_user_csv(num_hours=48)
        files = {"file": ("user_september.csv", csv_bytes, "text/csv")}
        response = self.client.post("/user-forecast/upload", files=files)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["accepted_rows"], 48)
        self.assertIn("session_id", data)
        self.assertIn("Univariate GRU", data["compatible_model"])
        self.assertIn("personal_analytics", data)
        analytics = data["personal_analytics"]
        self.assertGreater(analytics["total_consumption_kwh"], 0.0)
        self.assertGreater(analytics["average_hourly_kwh"], 0.0)
        self.assertEqual(len(data["preview_data"]), 10) # 5 head + 5 tail
        self.assertIn("recent_history", data)
        self.assertEqual(len(data["recent_history"]), 24)

    def test_03_predict_next_24_hours(self):
        """Test POST /user-forecast/predict generates 24-step forecast for user."""
        csv_bytes = generate_sample_user_csv(num_hours=48)
        upload_resp = self.client.post(
            "/user-forecast/upload",
            files={"file": ("user_data.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_resp.json()["session_id"]

        pred_resp = self.client.post(
            "/user-forecast/predict",
            json={"session_id": session_id, "horizon": "24h"}
        )
        self.assertEqual(pred_resp.status_code, 200)
        data = pred_resp.json()
        self.assertEqual(len(data["predictions"]), 24)
        self.assertEqual(data["horizon"], "24h")
        self.assertGreater(data["total_predicted_kwh"], 0.0)
        self.assertIn("peak_hour", data)
        self.assertIn("anomalies", data)
        self.assertIn("recommendations", data)
        self.assertGreater(len(data["recommendations"]), 0)
        for item in data["predictions"]:
            self.assertEqual(item["unit"], "kWh")
            self.assertAlmostEqual(item["predicted_wh"], item["predicted_consumption"] * 1000.0, places=1)

    def test_04_predict_next_hour(self):
        """Test POST /user-forecast/predict with 1h horizon."""
        csv_bytes = generate_sample_user_csv(num_hours=48)
        upload_resp = self.client.post(
            "/user-forecast/upload",
            files={"file": ("user_data.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_resp.json()["session_id"]

        pred_resp = self.client.post(
            "/user-forecast/predict",
            json={"session_id": session_id, "horizon": "1h"}
        )
        self.assertEqual(pred_resp.status_code, 200)
        data = pred_resp.json()
        self.assertEqual(len(data["predictions"]), 1)
        self.assertEqual(data["predictions"][0]["step"], 1)
        self.assertGreater(data["predictions"][0]["predicted_consumption"], 0.0)
        self.assertEqual(data["predictions"][0]["unit"], "kWh")
        self.assertAlmostEqual(data["predictions"][0]["predicted_wh"], data["predictions"][0]["predicted_consumption"] * 1000.0, places=1)

    def test_05_upload_small_gaps_interpolation(self):
        """Test small allowable gaps (<= 3 hours) are interpolated linearly."""
        csv_bytes = generate_sample_user_csv(num_hours=48, add_gaps=True)
        files = {"file": ("gapped_data.csv", csv_bytes, "text/csv")}
        response = self.client.post("/user-forecast/upload", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["is_valid"])
        self.assertEqual(data["accepted_rows"], 48)
        self.assertEqual(data["imputed_count"], 2)

    def test_06_reject_non_csv_file(self):
        """Test non-csv file extension is rejected with 422."""
        files = {"file": ("data.txt", b"plain text", "text/plain")}
        response = self.client.post("/user-forecast/upload", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIn("Only .csv files are supported", response.json()["detail"])

    def test_07_reject_insufficient_history(self):
        """Test file with fewer than 24 hours is rejected with 422."""
        csv_bytes = generate_sample_user_csv(num_hours=12) # only 12 hours
        files = {"file": ("short_data.csv", csv_bytes, "text/csv")}
        response = self.client.post("/user-forecast/upload", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIn("minimum 24 consecutive hourly observations required", response.json()["detail"])

    def test_08_reject_negative_readings(self):
        """Test negative energy readings are rejected with 422."""
        dates = pd.date_range(start="2026-09-01 00:00:00", periods=30, freq="1h")
        kwh = [1.5 if i != 5 else -2.3 for i in range(30)]
        df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"), "energy_consumption_kwh": kwh})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        files = {"file": ("negative_data.csv", buf.getvalue().encode(), "text/csv")}
        response = self.client.post("/user-forecast/upload", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIn("negative", response.json()["detail"].lower())

    def test_09_reject_excessive_contiguous_gap(self):
        """Test contiguous missing gap > 3 hours is rejected."""
        dates = pd.date_range(start="2026-09-01 00:00:00", periods=50, freq="1h")
        df = pd.DataFrame({"timestamp": dates.strftime("%Y-%m-%d %H:%M:%S"), "energy_consumption_kwh": 1.2})
        # Drop 5 contiguous hours (index 10 to 14)
        df = df.drop(index=list(range(10, 15))).reset_index(drop=True)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        files = {"file": ("gap_data.csv", buf.getvalue().encode(), "text/csv")}
        response = self.client.post("/user-forecast/upload", files=files)
        self.assertEqual(response.status_code, 422)
        self.assertIn("gap of 5 hours", response.json()["detail"])

    def test_10_missing_or_expired_session(self):
        """Test requesting forecast with invalid session returns 404."""
        response = self.client.post(
            "/user-forecast/predict",
            json={"session_id": "non-existent-session-id-12345", "horizon": "24h"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertIn("Session expired or not found", response.json()["detail"])

    def test_11_reset_session(self):
        """Test DELETE /user-forecast/session/{session_id} clears the session."""
        csv_bytes = generate_sample_user_csv(num_hours=30)
        upload_resp = self.client.post(
            "/user-forecast/upload",
            files={"file": ("user_data.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_resp.json()["session_id"]
        
        del_resp = self.client.delete(f"/user-forecast/session/{session_id}")
        self.assertEqual(del_resp.status_code, 200)
        self.assertTrue(del_resp.json()["cleared"])

        # Subsequent predict should return 404
        pred_resp = self.client.post(
            "/user-forecast/predict",
            json={"session_id": session_id, "horizon": "24h"}
        )
        self.assertEqual(pred_resp.status_code, 404)

    def test_12_regression_check_existing_endpoints(self):
        """Ensure original dashboard endpoints remain fully functional."""
        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        self.assertTrue(r_health.json()["model_loaded"])

        r_info = self.client.get("/model-info")
        self.assertEqual(r_info.status_code, 200)

        r_pred = self.client.post("/predict", json={})
        self.assertEqual(r_pred.status_code, 200)
        self.assertGreater(r_pred.json()["predicted_consumption"], 0.0)

    def test_13_diurnal_profile_and_historical_timeline(self):
        """Test upload and predict both return diurnal profile and historical timeline."""
        csv_bytes = generate_sample_user_csv(num_hours=48)
        upload_resp = self.client.post(
            "/user-forecast/upload",
            files={"file": ("user_timeline.csv", csv_bytes, "text/csv")}
        )
        self.assertEqual(upload_resp.status_code, 200)
        up_data = upload_resp.json()
        self.assertIn("diurnal_profile", up_data)
        self.assertEqual(len(up_data["diurnal_profile"]), 24)
        self.assertIn("historical_timeline", up_data)
        self.assertEqual(len(up_data["historical_timeline"]), 48)

        session_id = up_data["session_id"]
        pred_resp = self.client.post(
            "/user-forecast/predict",
            json={"session_id": session_id, "horizon": "24h"}
        )
        self.assertEqual(pred_resp.status_code, 200)
        pred_data = pred_resp.json()
        self.assertIn("diurnal_profile", pred_data)
        self.assertEqual(len(pred_data["diurnal_profile"]), 24)
        self.assertIn("historical_timeline", pred_data)
        self.assertEqual(len(pred_data["historical_timeline"]), 48)

    def test_14_structured_recommendations_and_anomalies(self):
        """Test recommendations and anomalies have explanation, severity, trigger patterns, and actions."""
        csv_bytes = generate_sample_user_csv(num_hours=48)
        upload_resp = self.client.post(
            "/user-forecast/upload",
            files={"file": ("user_anoms.csv", csv_bytes, "text/csv")}
        )
        session_id = upload_resp.json()["session_id"]
        pred_resp = self.client.post(
            "/user-forecast/predict",
            json={"session_id": session_id, "horizon": "24h"}
        )
        self.assertEqual(pred_resp.status_code, 200)
        data = pred_resp.json()

        # Check recommendations
        for rec in data["recommendations"]:
            self.assertIn("title", rec)
            self.assertIn("recommendation", rec)
            self.assertIn("trigger_pattern", rec)
            self.assertIn("action", rec)
            self.assertIn("estimated_savings", rec)
            self.assertTrue(rec["trigger_pattern"])
            self.assertTrue(rec["action"])
            self.assertTrue(rec["estimated_savings"])

        # Check anomalies
        for anom in data["anomalies"]["results"]:
            self.assertIn("severity", anom)
            self.assertIn("explanation", anom)
            self.assertIn(anom["severity"], ["Normal", "Moderate", "High"])

    def test_15_second_upload_session_isolation(self):
        """Test subsequent file upload generates unique session and previous session can be purged cleanly."""
        csv_1 = generate_sample_user_csv(num_hours=36, base_val=1.0)
        csv_2 = generate_sample_user_csv(num_hours=48, base_val=3.5)

        res_1 = self.client.post("/user-forecast/upload", files={"file": ("u1.csv", csv_1, "text/csv")}).json()
        res_2 = self.client.post("/user-forecast/upload", files={"file": ("u2.csv", csv_2, "text/csv")}).json()

        self.assertNotEqual(res_1["session_id"], res_2["session_id"])
        self.assertEqual(res_1["accepted_rows"], 36)
        self.assertEqual(res_2["accepted_rows"], 48)

        # Clear first session
        del_1 = self.client.delete(f"/user-forecast/session/{res_1['session_id']}")
        self.assertTrue(del_1.json()["cleared"])

        # Second session still valid and predicts correctly
        pred_2 = self.client.post("/user-forecast/predict", json={"session_id": res_2["session_id"], "horizon": "24h"})
        self.assertEqual(pred_2.status_code, 200)
        self.assertGreater(pred_2.json()["total_predicted_kwh"], 0)

if __name__ == "__main__":
    unittest.main()

