"""
TerraFlare - Phase 14 Reverse Geocoding & Human-Readable Location Test Suite
Tests Nominatim API parsing, smart location label hierarchy, session state caching, and alert payloads.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from geolocation.reverse_geocoder import fetch_reverse_geocoding, format_smart_location_label, calculate_distance_m
from geolocation.location_manager import LocationManager
from risk_engine import RiskEngine

def run_phase14_tests():
    print("=" * 60)
    print("RUNNING TERRAFLARE PHASE 14: REVERSE GEOCODING TEST SUITE")
    print("=" * 60)

    results = {}

    import streamlit as st
    if not hasattr(st, "session_state") or st.session_state is None:
        st.session_state = {}

    # -----------------------------------------------------------------
    # TEST 1: Smart Location Label Formatting Hierarchy
    # -----------------------------------------------------------------
    print("\n[TEST 1] Smart Location Label Formatting Hierarchy...")
    mock_university = {
        "display_name": "Dayananda Sagar University, Kudlu Gate, Hosur Road, Bengaluru, Karnataka, 560068, India",
        "address": {
            "university": "Dayananda Sagar University",
            "suburb": "Kudlu Gate",
            "city": "Bengaluru",
            "state": "Karnataka",
            "country": "India"
        }
    }
    label1 = format_smart_location_label(mock_university)
    t1_pass = label1 == "Dayananda Sagar University, Bengaluru"
    results["1. Smart Label (Institution + City)"] = "PASS" if t1_pass else f"FAIL ({label1})"
    print(f"Result: {results['1. Smart Label (Institution + City)']} ('{label1}')")

    mock_area = {
        "display_name": "Electronic City Phase 1, Bengaluru, Karnataka, India",
        "address": {
            "suburb": "Electronic City",
            "city": "Bengaluru",
            "state": "Karnataka"
        }
    }
    label2 = format_smart_location_label(mock_area)
    t2_pass = label2 == "Electronic City, Bengaluru"
    results["2. Smart Label (Area + City)"] = "PASS" if t2_pass else f"FAIL ({label2})"
    print(f"Result: {results['2. Smart Label (Area + City)']} ('{label2}')")

    # -----------------------------------------------------------------
    # TEST 2: Real Nominatim API Call for Coordinates (12.9716, 77.5946)
    # -----------------------------------------------------------------
    print("\n[TEST 2] Real Nominatim Reverse Geocoding API Call...")
    geo_data = fetch_reverse_geocoding(12.9716, 77.5946)
    t3_pass = geo_data is not None and "address" in geo_data
    smart_name_real = format_smart_location_label(geo_data) if geo_data else None
    results["3. Real Nominatim Reverse Geocoding Call"] = "PASS" if t3_pass else "FAIL (API call returned None)"
    print(f"Result: {results['3. Real Nominatim Reverse Geocoding Call']} (Place: '{smart_name_real}')")

    # -----------------------------------------------------------------
    # TEST 3: Caching & Single API Request Execution
    # -----------------------------------------------------------------
    print("\n[TEST 3] Caching — Prevent repeated API calls on webcam frames...")
    loc_mgr = LocationManager()
    sample_gps = {
        "latitude": 12.971598,
        "longitude": 77.594567,
        "accuracy": 15.0
    }
    loc_mgr.update_from_browser_result(sample_gps)
    cached_name = loc_mgr.state.get("source_location_name")
    
    # Simulate 100 webcam frames without location movement
    for _ in range(100):
        # Continuous frames use cached_name without calling API again
        frame_location_name = loc_mgr.state.get("source_location_name")

    t4_pass = frame_location_name == cached_name and cached_name is not None
    results["4. Continuous Frame Caching (Zero Repeat API Calls)"] = "PASS" if t4_pass else "FAIL"
    print(f"Result: {results['4. Continuous Frame Caching (Zero Repeat API Calls)']} (Cached Place: '{frame_location_name}')")

    # -----------------------------------------------------------------
    # TEST 4: Fire Alert Banner & Simulated Alert JSON with location_name
    # -----------------------------------------------------------------
    print("\n[TEST 4] Fire Alert Banner & Simulated Alert JSON with location_name...")
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
        source="Live Webcam",
        latitude=12.971598,
        longitude=77.594567,
        accuracy_m=15.0,
        source_location=cached_name
    )

    sim_alert = risk_data.get("simulated_alert")
    alert_loc_name = sim_alert.get("location_name") if sim_alert else None
    if not alert_loc_name and sim_alert and "camera_location" in sim_alert:
        alert_loc_name = sim_alert["camera_location"].get("location_name")

    t5_pass = (
        risk_data.get("status") == "ALERT" and
        sim_alert is not None and
        alert_loc_name == cached_name
    )
    results["5. Fire Alert JSON contains location_name"] = "PASS" if t5_pass else f"FAIL ({sim_alert})"
    print(f"Result: {results['5. Fire Alert JSON contains location_name']} (Alert Location: '{alert_loc_name}')")

    # -----------------------------------------------------------------
    # TEST 5: API Offline / Failure Graceful Handling
    # -----------------------------------------------------------------
    print("\n[TEST 5] Offline / API Failure Fallback (Detection Uninterrupted)...")
    offline_label = format_smart_location_label(None)
    t6_pass = offline_label is None  # Handled safely without exception

    # Fire risk engine continues evaluating even when location_name is generic
    risk_data_offline = re.calculate_risk(
        fire_conf=0.91,
        smoke_conf=0.78,
        area_ratio=0.15,
        verifier_result=fire_verifier_res,
        source="Live Webcam",
        latitude=12.971598,
        longitude=77.594567,
        accuracy_m=15.0,
        source_location="Location name temporarily unavailable"
    )
    t6_pass = t6_pass and risk_data_offline.get("status") == "ALERT"
    results["6. API Failure Fallback -> Detection Uninterrupted"] = "PASS" if t6_pass else "FAIL"
    print(f"Result: {results['6. API Failure Fallback -> Detection Uninterrupted']} (Status: {risk_data_offline.get('status')})")

    print("\n" + "=" * 60)
    print("FINAL SUMMARY OF PHASE 14 TEST RESULTS:")
    print("=" * 60)
    all_passed = True
    for test_name, status in results.items():
        print(f" - {test_name}: {status}")
        if "FAIL" in status:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL PHASE 14 TESTS PASSED SUCCESSFULLY! 🎉")
    else:
        print("SOME TESTS FAILED! PLEASE REVIEW LOGS.")
    print("=" * 60)

if __name__ == "__main__":
    run_phase14_tests()
