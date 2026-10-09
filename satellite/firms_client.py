"""
TerraFlare - Phase 15: NASA FIRMS API Client
Queries NASA FIRMS Active Fire data using the official area query CSV endpoint.
Endpoint: https://firms.modaps.eosdis.nasa.gov/api/area/csv/{MAP_KEY}/{SOURCE}/{BBOX}/{DAY_RANGE}
Handles MAP_KEY validation, bounding box calculation, HTTP error handling, and CSV parsing.
"""

import os
import csv
import math
import urllib.request
import urllib.error
from io import StringIO
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv

# Load .env if present
load_dotenv()

PREFERRED_SOURCES = [
    "VIIRS_NOAA21_NRT",
    "VIIRS_NOAA20_NRT",
    "MODIS_NRT",
    "VIIRS_SNPP_NRT"
]

SATELLITE_NAME_MAP = {
    "21": "VIIRS NOAA-21",
    "N21": "VIIRS NOAA-21",
    "NOAA-21": "VIIRS NOAA-21",
    "VIIRS_NOAA21_NRT": "VIIRS NOAA-21",
    "20": "VIIRS NOAA-20",
    "N20": "VIIRS NOAA-20",
    "NOAA-20": "VIIRS NOAA-20",
    "VIIRS_NOAA20_NRT": "VIIRS NOAA-20",
    "N": "VIIRS Suomi-NPP",
    "NPP": "VIIRS Suomi-NPP",
    "SNPP": "VIIRS Suomi-NPP",
    "VIIRS_SNPP_NRT": "VIIRS Suomi-NPP",
    "T": "MODIS (Terra)",
    "A": "MODIS (Aqua)",
    "Terra": "MODIS (Terra)",
    "Aqua": "MODIS (Aqua)",
    "MODIS_NRT": "MODIS"
}

def get_firms_map_key() -> Optional[str]:
    """Retrieves FIRMS_MAP_KEY from environment variables without exposing or logging it."""
    key = os.environ.get("FIRMS_MAP_KEY", "").strip()
    return key if key else None

def compute_bounding_box(lat: float, lon: float, radius_km: float) -> str:
    """
    Computes a Bounding Box string (west,south,east,north) centered at (lat, lon) with given radius in km.
    """
    delta_lat = radius_km / 111.32
    cos_lat = max(0.0001, math.cos(math.radians(lat)))
    delta_lon = radius_km / (111.32 * cos_lat)

    west = max(-180.0, lon - delta_lon)
    south = max(-90.0, lat - delta_lat)
    east = min(180.0, lon + delta_lon)
    north = min(90.0, lat + delta_lat)

    return f"{west:.4f},{south:.4f},{east:.4f},{north:.4f}"

def parse_acq_time(acq_time_str: str) -> str:
    """Formats 3-4 digit HHMM acquisition time string into 'HH:MM UTC'."""
    clean = str(acq_time_str).strip().zfill(4)
    if len(clean) >= 4:
        return f"{clean[:2]}:{clean[2:4]} UTC"
    return f"{clean} UTC"

def normalize_satellite_name(sat_code: str, source_product: str) -> str:
    """Maps raw satellite code or source product string to clean human-readable name."""
    clean_code = str(sat_code).strip()
    if clean_code in SATELLITE_NAME_MAP:
        return SATELLITE_NAME_MAP[clean_code]
    clean_src = str(source_product).strip()
    if clean_src in SATELLITE_NAME_MAP:
        return SATELLITE_NAME_MAP[clean_src]
    return f"Satellite ({clean_code or source_product})"

def parse_firms_csv(csv_text: str, source_product: str) -> List[Dict]:
    """Parses NASA FIRMS CSV output into structured hotspot dictionaries."""
    hotspots = []
    if not csv_text or "latitude" not in csv_text.lower():
        return hotspots

    f = StringIO(csv_text)
    reader = csv.DictReader(f)
    for row in reader:
        try:
            lat_val = float(row.get("latitude"))
            lon_val = float(row.get("longitude"))
            acq_date = row.get("acq_date", "").strip()
            acq_time_raw = row.get("acq_time", "").strip()
            acq_time_formatted = parse_acq_time(acq_time_raw)
            
            raw_sat = row.get("satellite", "").strip()
            sat_name = normalize_satellite_name(raw_sat, source_product)
            instrument = row.get("instrument", "VIIRS" if "VIIRS" in source_product else "MODIS").strip()
            
            confidence = row.get("confidence", "").strip()
            if confidence == "n":
                confidence = "nominal"
            elif confidence == "l":
                confidence = "low"
            elif confidence == "h":
                confidence = "high"

            frp_raw = row.get("frp")
            frp_val = float(frp_raw) if frp_raw is not None and str(frp_raw).strip() not in ("", "N/A", "None") else None

            hotspot = {
                "latitude": lat_val,
                "longitude": lon_val,
                "acq_date": acq_date,
                "acq_time": acq_time_formatted,
                "acq_time_raw": acq_time_raw,
                "satellite": sat_name,
                "instrument": instrument,
                "confidence": confidence if confidence else None,
                "frp": frp_val,
                "source_product": source_product
            }
            hotspots.append(hotspot)
        except (ValueError, TypeError, KeyError) as e:
            continue
    return hotspots

def fetch_firms_hotspots(
    lat: float,
    lon: float,
    radius_km: float = 5.0,
    time_window_hours: int = 24,
    map_key: Optional[str] = None
) -> Tuple[List[Dict], Optional[str], str]:
    """
    Fetches recent active-fire hotspots around (lat, lon) using NASA FIRMS API.
    Returns: (hotspots_list, error_message, status)
    status can be: "SUCCESS", "NO_MAP_KEY", "API_ERROR"
    """
    active_key = map_key or get_firms_map_key()
    if not active_key:
        return [], "NASA FIRMS MAP_KEY not configured.", "NO_MAP_KEY"

    bbox = compute_bounding_box(lat, lon, radius_km)
    day_range = 1 if time_window_hours <= 24 else min(10, math.ceil(time_window_hours / 24))

    all_hotspots = []
    errors = []

    for source in PREFERRED_SOURCES:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{active_key}/{source}/{bbox}/{day_range}"
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "TerraFlare-Wildfire-Verification/1.0"}
            )
            with urllib.request.urlopen(req, timeout=8) as response:
                if response.status == 200:
                    csv_text = response.read().decode("utf-8")
                    if "Invalid MAP_KEY" in csv_text or "map key not found" in csv_text.lower():
                        return [], "NASA FIRMS MAP_KEY is invalid or unauthorized.", "API_ERROR"
                    parsed = parse_firms_csv(csv_text, source)
                    all_hotspots.extend(parsed)
                else:
                    errors.append(f"{source}: HTTP {response.status}")
        except urllib.error.HTTPError as e:
            errors.append(f"{source}: HTTP {e.code}")
        except urllib.error.URLError as e:
            errors.append(f"{source}: Network Error ({e.reason})")
        except Exception as e:
            errors.append(f"{source}: {str(e)}")

    if not all_hotspots and errors and len(errors) == len(PREFERRED_SOURCES):
        return [], f"FIRMS API connection failed ({errors[0]})", "API_ERROR"

    return all_hotspots, None, "SUCCESS"
