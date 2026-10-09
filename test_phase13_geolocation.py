"""
TerraFlare - Phase 13 Real Browser Geolocation Test Suite (streamlit-geolocation package)
Tests browser GPS location manager states, permission states, unrounded metrics, HTTPS notices, and detection events.
"""

import sys
import os

# Ensure root workspace is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geolocation.location_manager import LocationManager
from geolocation.location_event import create_detection_event, generate_webcam_simulated_alert
from risk_engine import RiskEngine

def run_phase13_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 13: STREAMLIT-GEOLOCATION TEST SUITE")
    print("=" * 60)

    results = {}

    # Mocking Streamlit session state for test environment
    import streamlit as st
    if not hasattr(st, "session_state") or st.session_state is None:
        st.session_state = {}

    # -----------------------------------------------------------------
    # TEST 1: Allow Location -> Receive Raw Unrounded GPS Coordinates
    # -----------------------------------------------------------------
    print("\n[TEST 1] Allow Location -> Receive Raw Unrounded GPS Coordinates...")
    loc_mgr = LocationManager()
    sample_gps = {
        "latitude": 12.9715987654321,
        "longitude": 77.5945678901234,
        "accuracy": 12.34
    }
    loc_mgr.update_from_browser_result(sample_gps)
    state = loc_mgr.state

    t1_pass = (
        state["status"] == "GRANTED" and
        state["latitude"] == 12.9715987654321 and
        state["longitude"] == 77.5945678901234 and
        state["accuracy"] == 12.34 and
        state["source_type"] == "LIVE_BROWSER_GEOLOCATION"
    )
    results["1. Allow Location -> Raw Unrounded GPS Received"] = "PASS" if t1_pass else f"FAIL ({state})"
    print(f"Result: {results['1. Allow Location -> Raw Unrounded GPS Received']} (Lat: {state['latitude']}, Lon: {state['longitude']}, Acc: ±{state['accuracy']}m)")

    # -----------------------------------------------------------------
    # TEST 2: Fire Detected -> Detection Event & Simulated Alert Created
    # -----------------------------------------------------------------
    print("\n[TEST 2] Fire Detected -> Detection Event & Simulated Alert Created...")
    re = RiskEngine()
    fire_verifier_res = {
        "available": True,
        "status": "ACTIVE",
        "result": "FIRE",
        "confidence": 0.95,
        "verified": True,
        "verifier_score": 0.95,
        "suppression_factor": 1.0,
        "message": "Verified as FIRE (95.0%)"
    }
    risk_data = re.calculate_risk(
        fire_conf=0.91,
        smoke_conf=0.78,
        area_ratio=0.15,
        verifier_result=fire_verifier_res,
        source="Live Camera Feed",
        latitude=state["latitude"],
        longitude=state["longitude"],
        accuracy_m=state["accuracy"],
        location_source_type=state["source_type"]
    )
    
    det_evt = risk_data.get("detection_event")
    sim_alert = risk_data.get("simulated_alert")

    t2_pass = (
        risk_data.get("status") == "ALERT" and
        det_evt is not None and
        det_evt.get("source_label") == "SOURCE/CAMERA LOCATION" and
        det_evt.get("latitude") == 12.9715987654321 and
        det_evt.get("location_accuracy_m") == 12.34 and
        sim_alert is not None and
        sim_alert.get("alert_id").startswith("VR-ALERT-")
    )
    results["2. Fire Detected -> Camera Location Event Created"] = "PASS" if t2_pass else f"FAIL ({risk_data})"
    print(f"Result: {results['2. Fire Detected -> Camera Location Event Created']} (Event ID: {det_evt.get('event_id') if det_evt else 'None'})")

    # -----------------------------------------------------------------
    # TEST 3: No Fire -> Camera Location Preserved in NORMAL Status
    # -----------------------------------------------------------------
    print("\n[TEST 3] No Fire -> Camera Location Preserved in NORMAL Status...")
    risk_data_normal = re.calculate_risk(
        fire_conf=0.0,
        smoke_conf=0.0,
        area_ratio=0.0,
        source="Live Camera Feed",
        latitude=state["latitude"],
        longitude=state["longitude"],
        accuracy_m=state["accuracy"],
        location_source_type=state["source_type"]
    )
    t3_pass = (
        risk_data_normal.get("status") == "NORMAL" and
        risk_data_normal.get("latitude") == 12.9715987654321 and
        risk_data_normal.get("simulated_alert") is None
    )
    results["3. No Fire -> Camera Location Preserved in NORMAL Status"] = "PASS" if t3_pass else f"FAIL ({risk_data_normal})"
    print(f"Result: {results['3. No Fire -> Camera Location Preserved in NORMAL Status']} (Status: {risk_data_normal.get('status')})")

    # -----------------------------------------------------------------
    # TEST 4: Deny Location -> DENIED Status
    # -----------------------------------------------------------------
    print("\n[TEST 4] Deny Location -> DENIED Status...")
    denied_gps = {
        "status": "DENIED",
        "latitude": None,
        "longitude": None,
        "error_message": "Location permission denied or position unavailable."
    }
    loc_mgr.update_from_browser_result(denied_gps)
    state_denied = loc_mgr.state

    t4_pass = (
        state_denied["status"] == "DENIED" and
        state_denied["latitude"] is None and
        state_denied["longitude"] is None
    )
    results["4. Deny Location -> DENIED Status Handled"] = "PASS" if t4_pass else f"FAIL ({state_denied})"
    print(f"Result: {results['4. Deny Location -> DENIED Status Handled']} (Status: {state_denied.get('status')})")

    # -----------------------------------------------------------------
    # TEST 5: Geolocation Unavailable -> Configured Fallback
    # -----------------------------------------------------------------
    print("\n[TEST 5] Geolocation Unavailable -> Configured Fallback...")
    loc_mgr.set_configured_fallback(lat=12.9716, lon=77.5946, name="Configured Demo Location")
    state_fallback = loc_mgr.state

    t5_pass = (
        state_fallback["status"] == "FALLBACK" and
        state_fallback["source_type"] == "CONFIGURED_FALLBACK" and
        state_fallback["latitude"] == 12.9716 and
        state_fallback["source_location_name"] == "Configured Demo Location"
    )
    results["5. Geolocation Unavailable -> Configured Fallback Active"] = "PASS" if t5_pass else f"FAIL ({state_fallback})"
    print(f"Result: {results['5. Geolocation Unavailable -> Configured Fallback Active']} (Source Type: {state_fallback['source_type']})")

    # -----------------------------------------------------------------
    # TEST 6: Low GPS Accuracy -> Preserves Actual Accuracy Value
    # -----------------------------------------------------------------
    print("\n[TEST 6] Low GPS Accuracy -> Preserves Actual Accuracy Metric...")
    low_acc_gps = {
        "latitude": 40.712776,
        "longitude": -74.005974,
        "accuracy": 450.75
    }
    loc_mgr.update_from_browser_result(low_acc_gps)
    state_low_acc = loc_mgr.state

    t6_pass = (
        state_low_acc["status"] == "GRANTED" and
        state_low_acc["accuracy"] == 450.75
    )
    results["6. Low Accuracy -> Preserves Actual Accuracy Metric"] = "PASS" if t6_pass else f"FAIL ({state_low_acc})"
    print(f"Result: {results['6. Low Accuracy -> Preserves Actual Accuracy Metric']} (Accuracy: ±{state_low_acc['accuracy']}m)")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 13 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for test_name, status in results.items():
        print(f" - {test_name}: {status}")
        if "FAIL" in status:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 13 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase13_tests()
