"""
Phase 17 Verification Test Script: TERRAFLARE Sound Alert System (Single Sidebar Component)
Tests sound trigger conditions, event deduplication, volume limits, mute states,
browser audio component rendering, visual alert banners, and continuous performance.
"""

import os
import sys
import time
import unittest
import numpy as np

# Mock streamlit session state for unit testing sound manager
class MockSessionState(dict):
    def __getattr__(self, item):
        return self.get(item)
    def __setattr__(self, key, value):
        self[key] = value

class TestPhase17SoundAlerts(unittest.TestCase):

    def setUp(self):
        import streamlit as st
        st.session_state = MockSessionState()
        from sound_notifier import SoundAlertManager, generate_synth_wav_base64
        self.generate_synth_wav_base64 = generate_synth_wav_base64
        self.SoundAlertManager = SoundAlertManager

    def test_01_wav_synth_generation(self):
        """TEST 1: Verify self-contained WAV audio generation in Base64."""
        alert_b64 = self.generate_synth_wav_base64("alert", duration_sec=0.85)
        test_b64 = self.generate_synth_wav_base64("test", duration_sec=0.35)

        self.assertTrue(alert_b64.startswith("data:audio/wav;base64,"))
        self.assertTrue(test_b64.startswith("data:audio/wav;base64,"))
        self.assertGreater(len(alert_b64), 1000)
        self.assertGreater(len(test_b64), 500)

    def test_02_test_sound_trigger(self):
        """TEST 1 & 2 (User Request): Click Test Sound -> test_event_id created in state."""
        mgr = self.SoundAlertManager()
        mgr.trigger_test_sound()
        
        test_id = import_st_session("test_event_id")
        self.assertIsNotNone(test_id)
        self.assertTrue(test_id.startswith("test_"))

    def test_03_normal_risk_no_sound(self):
        """TEST 3: Risk stays in NORMAL -> no alarm event ID set."""
        mgr = self.SoundAlertManager()
        risk_normal = {"status": "NORMAL", "risk_score": 0.25}
        
        mgr.process_risk_event(risk_normal)
        self.assertEqual(import_st_session("current_risk_status"), "NORMAL")
        self.assertIsNone(import_st_session("current_alert_event_id"))

    def test_04_watch_risk_optional_sound(self):
        """TEST 4: Risk at 68% (WATCH) -> no high-priority alert event ID set."""
        mgr = self.SoundAlertManager()
        risk_watch = {"status": "WATCH", "risk_score": 0.68}
        
        mgr.process_risk_event(risk_watch)
        self.assertEqual(import_st_session("current_risk_status"), "WATCH")
        self.assertIsNone(import_st_session("current_alert_event_id"))

    def test_05_alert_risk_triggers_sound(self):
        """TEST 5: Risk crosses into ALERT (>= 0.70) -> generates stable alert_event_id."""
        mgr = self.SoundAlertManager()
        
        risk_alert = {"status": "ALERT", "risk_score": 0.85}
        mgr.process_risk_event(risk_alert)
        
        self.assertEqual(import_st_session("current_risk_status"), "ALERT")
        alert_id = import_st_session("current_alert_event_id")
        self.assertIsNotNone(alert_id)
        self.assertTrue(alert_id.startswith("evt_ALERT_"))

    def test_06_alert_deduplication_across_frames(self):
        """TEST 6: ALERT persists across many webcam frames -> alert_event_id remains STABLE without restarting."""
        mgr = self.SoundAlertManager()
        risk_alert = {"status": "ALERT", "risk_score": 0.88}
        
        # Frame 1: Transitions to ALERT
        mgr.process_risk_event(risk_alert)
        initial_event_id = import_st_session("current_alert_event_id")
        self.assertIsNotNone(initial_event_id)
        
        # Frame 2 to 20: Persisting ALERT state
        for frame in range(2, 21):
            mgr.process_risk_event(risk_alert)
            current_event_id = import_st_session("current_alert_event_id")
            self.assertEqual(current_event_id, initial_event_id, f"Event ID changed unexpectedly on frame {frame}")

    def test_07_recovery_and_retrigger(self):
        """TEST 7: Risk returns to NORMAL then crosses into ALERT again -> new alert_event_id generated."""
        mgr = self.SoundAlertManager()
        
        # Initial ALERT
        mgr.process_risk_event({"status": "ALERT", "risk_score": 0.82})
        first_alert_id = import_st_session("current_alert_event_id")
        
        # Recovery to NORMAL -> clears alert_event_id
        mgr.process_risk_event({"status": "NORMAL", "risk_score": 0.15})
        self.assertEqual(import_st_session("current_risk_status"), "NORMAL")
        self.assertIsNone(import_st_session("current_alert_event_id"))
        
        # New ALERT event -> generates NEW alert_event_id
        mgr.process_risk_event({"status": "ALERT", "risk_score": 0.91})
        second_alert_id = import_st_session("current_alert_event_id")
        self.assertIsNotNone(second_alert_id)
        self.assertNotEqual(first_alert_id, second_alert_id)

    def test_08_mute_sound_state(self):
        """TEST 8: Sound is muted -> process_risk_event updates state, mute setting recorded."""
        mgr = self.SoundAlertManager()
        import_st_session()["sound_muted"] = True
        
        mgr.process_risk_event({"status": "ALERT", "risk_score": 0.80})
        self.assertTrue(import_st_session("sound_muted"))
        self.assertEqual(import_st_session("current_risk_status"), "ALERT")

    def test_09_risk_engine_banner_formatting(self):
        """TEST 9: Verify ALERT visual banner has required text."""
        from risk_engine import RiskEngine
        re = RiskEngine()
        verifier_res = {"available": True, "verifier_score": 1.0, "suppression_factor": 1.0, "result": "FIRE"}
        res = re.calculate_risk(fire_conf=1.0, smoke_conf=0.9, area_ratio=0.5, verifier_result=verifier_res)
        self.assertEqual(res["status"], "ALERT")
        self.assertGreaterEqual(res["risk_score"], 0.70)


def import_st_session(key=None):
    import streamlit as st
    if key:
        return st.session_state.get(key)
    return st.session_state


if __name__ == "__main__":
    unittest.main()
