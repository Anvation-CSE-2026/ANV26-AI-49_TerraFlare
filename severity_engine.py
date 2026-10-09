"""
TerraFlare - Phase 2 Severity Classification & Simulated Alert Module
Categorizes wildfire threat severity and generates structured simulated emergency alerts with geospatial location.
"""

from datetime import datetime
import time

def classify_severity(risk_score: float) -> dict:
    """
    Classifies wildfire severity based on Phase 1 Risk Score.
    - Risk < 0.40     -> LOW
    - Risk 0.40–0.69  -> MEDIUM
    - Risk >= 0.70    -> HIGH
    """
    score = float(risk_score)
    if score < 0.40:
        return {
            "severity": "LOW",
            "color": "#10B981",  # Emerald Green
            "icon": "🟢",
            "description": "Low wildfire threat. Detections are minimal or absent."
        }
    elif score < 0.70:
        return {
            "severity": "MEDIUM",
            "color": "#F59E0B",  # Amber / Yellow
            "icon": "🟡",
            "description": "Moderate wildfire threat. Active monitoring recommended."
        }
    else:
        return {
            "severity": "HIGH",
            "color": "#EF4444",  # Crimson Red
            "icon": "🔴",
            "description": "High wildfire threat! Simulated alert threshold exceeded."
        }


def generate_simulated_alert(
    risk_score: float,
    fire_conf: float,
    smoke_conf: float,
    area_ratio: float,
    source: str = "Image",
    latitude: float = 12.9716,
    longitude: float = 77.5946,
    source_location: str = "Bengaluru Demo Location"
) -> dict:
    """
    Generates a structured simulated forest fire alert payload when risk_score >= 0.70.
    IMPORTANT: This is a simulation payload for judging/demonstration only.
    No real emergency notifications or external agency connections are made.
    """
    severity_info = classify_severity(risk_score)
    now_iso = datetime.now().astimezone().isoformat()
    timestamp_id = int(time.time() * 1000) % 100000

    alert_payload = {
        "alert_id": f"TF-2026-{timestamp_id:05d}",
        "alert_type": "SIMULATED_FOREST_FIRE_ALERT",
        "timestamp": now_iso,
        "status": "ALERT",
        "severity": severity_info["severity"],
        "risk_score": round(float(risk_score), 4),
        "fire_confidence": round(float(fire_conf), 4),
        "smoke_confidence": round(float(smoke_conf), 4),
        "detection_area_ratio": round(float(area_ratio), 4),
        "source": source,
        "source_location": source_location,
        "latitude": round(float(latitude), 4),
        "longitude": round(float(longitude), 4),
        "simulation_notice": "THIS IS A SIMULATED ALERT FOR DEMONSTRATION/JUDGING PURPOSES. NO EMERGENCY SERVICES WERE CONTACTED."
    }

    return alert_payload
