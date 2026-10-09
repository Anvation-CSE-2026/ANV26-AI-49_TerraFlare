"""
TerraFlare - Phase 15: Satellite Data Models
Defines structured objects and schema builders for satellite cross-verification.
"""

def create_empty_satellite_verification(
    status: str = "UNAVAILABLE",
    status_message: str = "NASA FIRMS MAP_KEY not configured.",
    search_radius_km: float = 5.0,
    time_window_hours: int = 24
) -> dict:
    """
    Returns a standardized satellite verification dictionary when FIRMS is unavailable or not queried.
    """
    return {
        "status": status,  # "CORROBORATED", "NOT_CORROBORATED", "UNAVAILABLE"
        "status_message": status_message,
        "source": "NASA FIRMS",
        "satellite": None,
        "latitude": None,
        "longitude": None,
        "distance_km": None,
        "acquisition_date": None,
        "acquisition_time": None,
        "confidence": None,
        "frp": None,
        "search_radius_km": search_radius_km,
        "time_window_hours": time_window_hours,
        "hotspot_count": 0,
        "all_nearby_hotspots": []
    }

def format_satellite_alert_payload(sat_verification: dict) -> dict:
    """
    Formats satellite verification data specifically for inclusion in the simulated alert JSON payload.
    Only includes fields actually present; preserves null for missing fields without inventing values.
    """
    if not sat_verification or not isinstance(sat_verification, dict):
        return {
            "status": "UNAVAILABLE",
            "source": "NASA FIRMS",
            "status_message": "Satellite verification data missing"
        }

    status = sat_verification.get("status", "UNAVAILABLE")
    
    payload = {
        "status": status,
        "source": sat_verification.get("source", "NASA FIRMS")
    }

    if status == "CORROBORATED":
        payload.update({
            "satellite": sat_verification.get("satellite"),
            "latitude": sat_verification.get("latitude"),
            "longitude": sat_verification.get("longitude"),
            "distance_km": sat_verification.get("distance_km"),
            "acquisition_date": sat_verification.get("acquisition_date"),
            "acquisition_time": sat_verification.get("acquisition_time"),
            "confidence": sat_verification.get("confidence"),
            "frp": sat_verification.get("frp")
        })
    elif status == "NOT_CORROBORATED":
        payload.update({
            "status_message": sat_verification.get("status_message", "No active FIRMS satellite hotspots detected within radius."),
            "search_radius_km": sat_verification.get("search_radius_km"),
            "time_window_hours": sat_verification.get("time_window_hours")
        })
    else:  # UNAVAILABLE
        payload.update({
            "status_message": sat_verification.get("status_message", "NASA FIRMS MAP_KEY not configured or API unavailable.")
        })

    return payload
