"""
TerraFlare - Phase 14: Reverse Geocoding Module
Converts GPS coordinates (lat, lon) into concise, human-readable place labels via OpenStreetMap Nominatim API.
Includes smart label formatting, caching, and fallback handling.
"""

import urllib.request
import urllib.parse
import json
import math

USER_AGENT = "TerraFlare/1.0 (Real-Time AI Wildfire Detection System)"

def fetch_reverse_geocoding(lat: float, lon: float, timeout: int = 4) -> dict:
    """
    Calls OpenStreetMap Nominatim Reverse Geocoding API.
    Returns JSON payload dictionary or None if request fails or times out.
    """
    if lat is None or lon is None:
        return None

    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=jsonv2&zoom=18&addressdetails=1"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT}
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                data = json.loads(response.read().decode('utf-8'))
                return data
    except Exception as e:
        print(f"[ReverseGeocoder] Nominatim reverse geocoding failed: {e}")
        return None

def format_smart_location_label(data: dict) -> str:
    """
    Parses Nominatim JSON dictionary to construct a concise, smart place label.
    Examples:
    - 'Dayananda Sagar University, Bengaluru'
    - 'Electronic City, Bengaluru'
    - 'Jayanagar, Bengaluru'
    - 'Bengaluru, Karnataka'
    """
    if not data or not isinstance(data, dict):
        return None

    address = data.get("address", {})
    display_name = data.get("display_name", "")

    # Extract landmark / institution / amenity
    building_landmark = (
        address.get("amenity") or
        address.get("university") or
        address.get("college") or
        address.get("school") or
        address.get("hospital") or
        address.get("building") or
        address.get("tourism") or
        address.get("historic") or
        address.get("leisure") or
        address.get("shop")
    )

    # Extract locality / area
    neighbourhood = (
        address.get("neighbourhood") or
        address.get("suburb") or
        address.get("residential") or
        address.get("quarter") or
        address.get("industrial")
    )

    # Extract city / town
    city = (
        address.get("city") or
        address.get("town") or
        address.get("municipality") or
        address.get("village") or
        address.get("county") or
        address.get("district")
    )

    state = address.get("state")

    # Smart hierarchy selection
    if building_landmark and city:
        return f"{building_landmark}, {city}"
    elif building_landmark and state:
        return f"{building_landmark}, {state}"
    elif neighbourhood and city:
        return f"{neighbourhood}, {city}"
    elif city and state:
        return f"{city}, {state}"
    elif building_landmark:
        return building_landmark
    elif city:
        return city
    elif display_name:
        parts = [p.strip() for p in display_name.split(",")]
        if len(parts) >= 2:
            return f"{parts[0]}, {parts[1]}"
        return parts[0]

    return None

def calculate_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes approximate distance in meters between two lat/lon points using Haversine formula.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 999999.0

    R = 6371000.0  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2) + math.cos(phi1) * math.cos(phi2) * (math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c
