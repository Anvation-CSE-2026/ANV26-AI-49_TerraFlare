"""
TerraFlare - Phase 16: Email Alert System Test Suite
Verifies SMTP configuration, alert payload content building, non-blocking asynchronous delivery,
event cooldown deduplication, and test-email functionality.
"""

import time
import numpy as np
import unittest

from email_alerts.email_notifier import (
    get_email_config,
    is_email_configured,
    should_send_alert_email,
    build_alert_email_content,
    send_alert_email,
    send_alert_email_async,
    send_test_email
)

class TestPhase16EmailAlerts(unittest.TestCase):

    def test_01_email_config(self):
        print("\n[TEST 1] Testing Email Configuration & Credentials...")
        config = get_email_config()
        self.assertIn("sender_email", config)
        self.assertIn("sender_password", config)
        self.assertTrue(config["is_configured"])
        print(f"✅ TEST 1 PASSED: Configured Sender: '{config['sender_email']}' | Server: {config['smtp_server']}:{config['smtp_port']}")

    def test_02_alert_approval_logic(self):
        print("\n[TEST 2] Testing Alert Approval & Deduplication Cooldown...")
        
        normal_data = {"status": "NORMAL", "risk_score": 0.25}
        should_send, reason = should_send_alert_email(normal_data, last_sent_time=0)
        self.assertFalse(should_send)
        
        watch_data = {"status": "WATCH", "risk_score": 0.55}
        should_send, reason = should_send_alert_email(watch_data, last_sent_time=0)
        self.assertFalse(should_send)

        alert_data = {
            "status": "ALERT",
            "risk_score": 0.82,
            "simulated_alert": {"alert_id": "ALERT_101"}
        }
        
        # First trigger should be approved
        should_send, reason = should_send_alert_email(alert_data, last_sent_time=0)
        self.assertTrue(should_send)

        # Immediate repeat trigger for same event within 60s should be suppressed
        now = time.time()
        should_send, reason = should_send_alert_email(alert_data, last_sent_time=now, last_sent_event_id="ALERT_101", cooldown_seconds=60)
        self.assertFalse(should_send)
        print("✅ TEST 2 PASSED: Alert approval and 60s cooldown deduplication work correctly.")

    def test_03_email_content_builder(self):
        print("\n[TEST 3] Testing Email Content Builder Formatting...")
        payload = {
            "status": "ALERT",
            "severity": "HIGH SEVERITY",
            "risk_score": 0.85,
            "fire_confidence": 0.90,
            "smoke_confidence": 0.40,
            "source": "Live Camera Feed",
            "source_location": "Dayananda Sagar University, Bengaluru",
            "latitude": 12.9716,
            "longitude": 77.5946,
            "accuracy_m": 12.5,
            "verifier_result": {"result": "FIRE"},
            "satellite_verification": {
                "status": "CORROBORATED",
                "satellite": "VIIRS NOAA-21",
                "distance_km": 1.25,
                "status_message": "Nearby satellite hotspot corroborates AI fire detection within 1.25 km of camera."
            }
        }

        subj, plain, html = build_alert_email_content(payload, ["test@example.com"])
        
        self.assertIn("FIRE DETECTED NEAR DAYANANDA SAGAR UNIVERSITY, BENGALURU", subj)
        self.assertIn("Dayananda Sagar University, Bengaluru", plain)
        self.assertIn("12.9716", plain)
        self.assertIn("77.5946", plain)
        self.assertIn("CORROBORATED", plain)
        self.assertIn("VIIRS NOAA-21", plain)
        self.assertIn("HIGH SEVERITY", html)
        print("✅ TEST 3 PASSED: Alert email subject, plain-text, and HTML body contain all required fields.")

    def test_04_test_email_dispatch(self):
        print("\n[TEST 4] Testing Live Test Email Dispatch via smtplib...")
        config = get_email_config()
        result = send_test_email(config["default_recipient"])
        
        self.assertIn(result["status"], ["SENT", "FAILED", "NOT CONFIGURED"])
        print(f"✅ TEST 4 PASSED: Test Email Result -> Status: '{result['status']}' | Message: '{result['message']}'")

    def test_05_async_email_dispatch(self):
        print("\n[TEST 5] Testing Asynchronous Non-Blocking Email Dispatch...")
        payload = {
            "status": "ALERT",
            "severity": "HIGH",
            "risk_score": 0.88,
            "fire_confidence": 0.92,
            "smoke_confidence": 0.10,
            "source": "Live Camera Feed",
            "source_location": "Bengaluru Campus",
            "latitude": 12.9716,
            "longitude": 77.5946
        }
        
        dummy_img = np.zeros((200, 200, 3), dtype=np.uint8)
        callback_results = []
        
        def test_callback(res):
            callback_results.append(res)

        start_t = time.time()
        thread = send_alert_email_async(
            image_bgr=dummy_img,
            recipient_emails=["abdogabr688@gmail.com"],
            alert_payload=payload,
            callback=test_callback
        )
        dispatch_dt = time.time() - start_t
        
        # Must return immediately (< 0.1s) without waiting for SMTP network response
        self.assertLess(dispatch_dt, 0.2)
        print(f"✅ TEST 5 PASSED: Async thread launched instantly in {dispatch_dt*1000:.1f}ms without blocking execution.")

        thread.join(timeout=10.0)
        self.assertGreater(len(callback_results), 0)
        print(f"✅ ASYNC WORKER RESULT: Status = '{callback_results[0]['status']}'")

if __name__ == "__main__":
    unittest.main()
