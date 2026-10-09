"""
TerraFlare - Phase 15: Satellite Hotspot Matcher
Calculates Haversine geographic distances between camera GPS coordinates and NASA FIRMS hotspots.
Selects the nearest relevant hotspot and computes satellite corroboration status according to the three states.
"""

import math
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional
from satellite.satellite_models import create_empty_satellite_verification

def calculate_haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates the great-circle distance between two points on the Earth using the Haversine formula.
    Returns distance in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2.0) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2.0) ** 2)
    
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return R * c

def filter_and_sort_hotspots(
    camera_lat: float,
    camera_lon: float,
    hotspots: List[Dict],
    search_radius_km: float = 5.0
) -> List[Dict]:
    """
    Calculates Haversine distance for each hotspot and returns those within search_radius_km, sorted by distance.
    """
    matched = []
    for hp in hotspots:
        try:
            h_lat = float(hp["latitude"])
            h_lon = float(hp["longitude"])
            dist_km = calculate_haversine_distance_km(camera_lat, camera_lon, h_lat, h_lon)
            
            if dist_km <= search_radius_km:
                hp_copy = dict(hp)
                hp_copy["distance_km"] = round(dist_km, 2)
                matched.append(hp_copy)
        except (KeyError, ValueError, TypeError):
            continue

    matched.sort(key=lambda x: x["distance_km"])
    return matched

def match_nearby_hotspots(
    camera_lat: Optional[float],
    camera_lon: Optional[float],
    hotspots: List[Dict],
    fetch_error: Optional[str] = None,
    fetch_status: str = "SUCCESS",
    search_radius_km: float = 5.0,
    time_window_hours: int = 24,
    ai_status: str = "ALERT",
    is_fire_event: bool = True
) -> Dict:
    """
    Matches NASA FIRMS hotspots with camera location and produces a structured satellite_verification object.
    
    States:
    1. CORROBORATED: AI detected fire/risk AND relevant FIRMS hotspot found within search_radius_km.
    2. NOT_CORROBORATED: AI detected fire/risk but no FIRMS hotspot found within radius/time window.
    3. UNAVAILABLE: API error, missing key, internet failure, or missing camera coordinates.
    """
    if camera_lat is None or camera_lon is None:
        return create_empty_satellite_verification(
            status="UNAVAILABLE",
            status_message="Camera GPS coordinates unavailable.",
            search_radius_km=search_radius_km,
            time_window_hours=time_window_hours
        )

    if fetch_status != "SUCCESS" or fetch_error is not None:
        msg = fetch_error or "NASA FIRMS MAP_KEY not configured or API unavailable."
        return create_empty_satellite_verification(
            status="UNAVAILABLE",
            status_message=msg,
            search_radius_km=search_radius_km,
            time_window_hours=time_window_hours
        )

    matched_hotspots = filter_and_sort_hotspots(camera_lat, camera_lon, hotspots, search_radius_km)

    if not is_fire_event:
        # Non-fire predictions (NORMAL/SAFE)
        if matched_hotspots:
            nearest = matched_hotspots[0]
            return {
                "status": "NOT_CORROBORATED",
                "status_message": f"Nearby satellite hotspot exists ({nearest['distance_km']} km away), but AI model detected non-fire/safe scene.",
                "source": "NASA FIRMS",
                "satellite": nearest.get("satellite"),
                "latitude": nearest.get("latitude"),
                "longitude": nearest.get("longitude"),
                "distance_km": nearest.get("distance_km"),
                "acquisition_date": nearest.get("acq_date"),
                "acquisition_time": nearest.get("acq_time"),
                "confidence": nearest.get("confidence"),
                "frp": nearest.get("frp"),
                "search_radius_km": search_radius_km,
                "time_window_hours": time_window_hours,
                "hotspot_count": len(matched_hotspots),
                "all_nearby_hotspots": matched_hotspots
            }
        else:
            return {
                "status": "NOT_CORROBORATED",
                "status_message": "Satellite verification not required for normal non-fire detections.",
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

    # Fire / Risk Event (ALERT or WATCH or Fire Detected)
    if matched_hotspots:
        nearest = matched_hotspots[0]
        return {
            "status": "CORROBORATED",
            "status_message": f"Satellite hotspot corroborates AI fire detection within {nearest['distance_km']} km of camera.",
            "source": "NASA FIRMS",
            "satellite": nearest.get("satellite"),
            "latitude": nearest.get("latitude"),
            "longitude": nearest.get("longitude"),
            "distance_km": nearest.get("distance_km"),
            "acquisition_date": nearest.get("acq_date"),
            "acquisition_time": nearest.get("acq_time"),
            "confidence": nearest.get("confidence"),
            "frp": nearest.get("frp"),
            "search_radius_km": search_radius_km,
            "time_window_hours": time_window_hours,
            "hotspot_count": len(matched_hotspots),
            "all_nearby_hotspots": matched_hotspots
        }
    else:
        return {
            "status": "NOT_CORROBORATED",
            "status_message": f"AI detected fire, but no relevant FIRMS hotspot was found within {search_radius_km} km in the last {time_window_hours}h. (This does NOT mean the AI detection is false; satellite latency or cloud cover may apply).",
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
