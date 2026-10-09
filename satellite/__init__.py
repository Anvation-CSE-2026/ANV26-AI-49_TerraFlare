"""
TerraFlare - Phase 15: NASA FIRMS Satellite Cross-Verification Package
Provides independent satellite active-fire corroboration using NASA FIRMS API.
"""

from satellite.satellite_models import create_empty_satellite_verification
from satellite.firms_client import fetch_firms_hotspots
from satellite.satellite_matcher import calculate_haversine_distance_km, match_nearby_hotspots
from satellite.satellite_cache import get_cached_satellite_verification, clear_satellite_cache
from satellite.satellite_ui import render_satellite_panel

__all__ = [
    "create_empty_satellite_verification",
    "fetch_firms_hotspots",
    "calculate_haversine_distance_km",
    "match_nearby_hotspots",
    "get_cached_satellite_verification",
    "clear_satellite_cache",
    "render_satellite_panel"
]
