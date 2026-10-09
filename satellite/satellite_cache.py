import time
import threading
import streamlit as st
from typing import Dict, Optional
from satellite.firms_client import fetch_firms_hotspots
from satellite.satellite_matcher import match_nearby_hotspots
from satellite.satellite_models import create_empty_satellite_verification

CACHE_TTL_SECONDS = 300.0  # 5 minutes cache lifetime
_ACTIVE_THREADS = {}  # In-memory registry of active background fetch threads

def _get_cache_key(lat: Optional[float], lon: Optional[float], radius_km: float, time_window_h: int, is_fire: bool) -> str:
    if lat is None or lon is None:
        return "invalid"
    return f"{round(lat, 3)}_{round(lon, 3)}_{radius_km}_{time_window_h}_{is_fire}"

def clear_satellite_cache():
    """Clears satellite verification cache in session state."""
    if "satellite_cache" in st.session_state:
        st.session_state.satellite_cache = {}

def _run_background_fetch(cache_key: str, lat: float, lon: float, radius_km: float, time_window_hours: int, ai_status: str, is_fire_event: bool, cache_ref: dict):
    """Worker function executed in a background thread for non-blocking FIRMS API calls."""
    try:
        hotspots, fetch_err, fetch_status = fetch_firms_hotspots(
            lat=lat,
            lon=lon,
            radius_km=radius_km,
            time_window_hours=time_window_hours
        )

        sat_verification = match_nearby_hotspots(
            camera_lat=lat,
            camera_lon=lon,
            hotspots=hotspots,
            fetch_error=fetch_err,
            fetch_status=fetch_status,
            search_radius_km=radius_km,
            time_window_hours=time_window_hours,
            ai_status=ai_status,
            is_fire_event=is_fire_event
        )

        cache_ref[cache_key] = {
            "timestamp": time.time(),
            "data": sat_verification
        }
    except Exception as e:
        cache_ref[cache_key] = {
            "timestamp": time.time(),
            "data": create_empty_satellite_verification(
                status="UNAVAILABLE",
                status_message=f"Background FIRMS fetch error: {e}",
                search_radius_km=radius_km,
                time_window_hours=time_window_hours
            )
        }
    finally:
        _ACTIVE_THREADS.pop(cache_key, None)

def get_cached_satellite_verification(
    camera_lat: Optional[float],
    camera_lon: Optional[float],
    search_radius_km: float = 5.0,
    time_window_hours: int = 24,
    ai_status: str = "ALERT",
    is_fire_event: bool = True,
    force_refresh: bool = False,
    is_async: bool = False
) -> Dict:
    """
    Fetches or returns cached satellite cross-verification results.
    If is_async is True (live stream mode), executes FIRMS API calls in a background thread
    so the live webcam inference loop is NEVER blocked.
    """
    if camera_lat is None or camera_lon is None:
        return create_empty_satellite_verification(
            status="UNAVAILABLE",
            status_message="Camera GPS coordinates not acquired.",
            search_radius_km=search_radius_km,
            time_window_hours=time_window_hours
        )

    if "satellite_cache" not in st.session_state:
        st.session_state.satellite_cache = {}

    cache_key = _get_cache_key(camera_lat, camera_lon, search_radius_km, time_window_hours, is_fire_event)
    now = time.time()

    # Return cached data if valid and not force_refresh
    if not force_refresh and cache_key in st.session_state.satellite_cache:
        entry = st.session_state.satellite_cache[cache_key]
        if (now - entry["timestamp"]) < CACHE_TTL_SECONDS:
            return entry["data"]

    # Non-blocking async mode for live webcam stream
    if is_async:
        # If thread is already querying for this cache_key in background, return requesting payload
        if cache_key in _ACTIVE_THREADS and _ACTIVE_THREADS[cache_key].is_alive():
            return {
                "status": "REQUESTING",
                "status_message": "Querying NASA FIRMS active-fire satellite data in background...",
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

        # Launch background worker thread
        bg_thread = threading.Thread(
            target=_run_background_fetch,
            args=(cache_key, camera_lat, camera_lon, search_radius_km, time_window_hours, ai_status, is_fire_event, st.session_state.satellite_cache),
            daemon=True
        )
        _ACTIVE_THREADS[cache_key] = bg_thread
        bg_thread.start()

        # If previous result exists in cache (e.g. force refresh), return it while thread updates
        if cache_key in st.session_state.satellite_cache:
            return st.session_state.satellite_cache[cache_key]["data"]

        return {
            "status": "REQUESTING",
            "status_message": "Querying NASA FIRMS active-fire satellite data in background...",
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

    # Synchronous mode (for static images or manual refresh)
    hotspots, fetch_err, fetch_status = fetch_firms_hotspots(
        lat=camera_lat,
        lon=camera_lon,
        radius_km=search_radius_km,
        time_window_hours=time_window_hours
    )

    sat_verification = match_nearby_hotspots(
        camera_lat=camera_lat,
        camera_lon=camera_lon,
        hotspots=hotspots,
        fetch_error=fetch_err,
        fetch_status=fetch_status,
        search_radius_km=search_radius_km,
        time_window_hours=time_window_hours,
        ai_status=ai_status,
        is_fire_event=is_fire_event
    )

    st.session_state.satellite_cache[cache_key] = {
        "timestamp": now,
        "data": sat_verification
    }

    return sat_verification
