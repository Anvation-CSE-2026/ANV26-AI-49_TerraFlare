"""
TerraFlare - Phase 14 Geolocation Package
"""

from geolocation.location_manager import LocationManager
from geolocation.location_event import create_detection_event, generate_webcam_simulated_alert
from geolocation.reverse_geocoder import fetch_reverse_geocoding, format_smart_location_label, calculate_distance_m

__all__ = [
    "LocationManager",
    "create_detection_event",
    "generate_webcam_simulated_alert",
    "fetch_reverse_geocoding",
    "format_smart_location_label",
    "calculate_distance_m"
]
