"""
TerraFlare - Phase 14: Detection Event & Alert Location Module
Creates structured detection events and simulated alert payloads using live browser coordinates and reverse-geocoded place names.
"""

import time
import random

def create_detection_event(
    risk_score: float,
    severity: str,
    detected_class: str,
    fire_confidence: float,
    smoke_confidence: float,
    latitude: float,
    longitude: float,
    accuracy: float,
    location_name: str = None,
    location_source_type: str = "LIVE_BROWSER_GEOLOCATION",
    source: str = "LIVE_WEBCAM",
    satellite_verification: dict = None
) -> dict:
    """
    Creates a structured fire detection event for ALERT status (Risk >= 0.70).
    Explicitly labels coordinates as the CAMERA/SOURCE LOCATION.
    """
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rand_suffix = random.randint(1000, 9999)
    event_id = f"VR-EVT-{time.strftime('%Y%m%d')}-{rand_suffix}"

    evt = {
        "event_id": event_id,
        "timestamp": now_str,
        "source": source,
        "source_label": "SOURCE/CAMERA LOCATION",
        "location_name": location_name or "Camera Location",
        "latitude": float(latitude) if latitude is not None else None,
        "longitude": float(longitude) if longitude is not None else None,
        "location_accuracy_m": float(accuracy) if accuracy is not None else None,
        "location_source_type": location_source_type,
        "risk_score": float(risk_score),
        "severity": severity,
        "detected_class": detected_class,
        "fire_confidence": float(fire_confidence),
        "smoke_confidence": float(smoke_confidence)
    }

    if satellite_verification and isinstance(satellite_verification, dict):
        from satellite.satellite_models import format_satellite_alert_payload
        evt["satellite_verification"] = format_satellite_alert_payload(satellite_verification)

    return evt

def generate_webcam_simulated_alert(
    risk_score: float,
    severity: str,
    detected_class: str,
    fire_confidence: float,
    smoke_confidence: float,
    latitude: float,
    longitude: float,
    accuracy: float,
    location_name: str = None,
    location_source_type: str = "LIVE_BROWSER_GEOLOCATION",
    source: str = "LIVE_WEBCAM",
    satellite_verification: dict = None
) -> dict:
    """
    Generates simulated alert JSON payload incorporating reverse-geocoded location_name, live coordinates, accuracy, and satellite verification.
    """
    now_str = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rand_suffix = random.randint(1000, 9999)
    alert_id = f"VR-ALERT-{time.strftime('%Y%m%d')}-{rand_suffix}"

    from satellite.satellite_models import format_satellite_alert_payload
    sat_payload = format_satellite_alert_payload(satellite_verification) if satellite_verification else {
        "status": "UNAVAILABLE",
        "source": "NASA FIRMS",
        "status_message": "NASA FIRMS MAP_KEY not configured or API unavailable."
    }

    return {
        "event_id": alert_id,
        "alert_id": alert_id,
        "timestamp": now_str,
        "source": source,
        "camera_location": {
            "latitude": round(float(latitude), 6) if latitude is not None else None,
            "longitude": round(float(longitude), 6) if longitude is not None else None,
            "accuracy_m": round(float(accuracy), 1) if accuracy is not None else None,
            "location_name": location_name or "Camera Location"
        },
        "ai_detection": {
            "detected_class": detected_class,
            "fire_confidence": round(float(fire_confidence), 4),
            "smoke_confidence": round(float(smoke_confidence), 4),
            "risk": round(float(risk_score), 4),
            "severity": severity,
            "status": "ALERT"
        },
        "satellite_verification": sat_payload,
        "notice": "SIMULATION NOTICE: This payload represents a simulated early-warning dispatch anchored to the camera location. No external emergency services are contacted."
    }

