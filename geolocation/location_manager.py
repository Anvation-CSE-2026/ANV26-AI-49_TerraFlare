"""
TerraFlare - Phase 14: Location Manager
Manages real browser geolocation state, reverse geocoding via Nominatim API, smart place label caching, and fallback rules.
"""

import time
import streamlit as st
from streamlit_geolocation import streamlit_geolocation
from geolocation.reverse_geocoder import fetch_reverse_geocoding, format_smart_location_label, calculate_distance_m

class LocationManager:
    """
    Stateful manager for live browser geolocation and reverse geocoding.
    Tracks status: 'NOT_SET', 'GRANTED', 'DENIED', 'FALLBACK'
    Caches reverse-geocoded place labels to avoid calling APIs during continuous YOLO inference frames.
    """
    DEFAULT_FALLBACK_LAT = 12.9716
    DEFAULT_FALLBACK_LON = 77.5946
    DEFAULT_FALLBACK_NAME = "Bengaluru Demo Location"
    DEFAULT_FALLBACK_ACCURACY = 500.0  # meters

    def __init__(self):
        if "geo_location_state" not in st.session_state:
            st.session_state.geo_location_state = {
                "status": "NOT_SET",
                "latitude": None,
                "longitude": None,
                "accuracy": None,
                "source_type": "NOT_SET",
                "source_location_name": None,
                "full_address_data": None,
                "last_geocoded_lat": None,
                "last_geocoded_lon": None,
                "timestamp": None,
                "error_message": None
            }

    @property
    def state(self) -> dict:
        return st.session_state.geo_location_state

    def reverse_geocode_current_location(self, force: bool = False):
        """
        Reverse geocodes current coordinates using Nominatim API ONLY when needed:
        - Never geocoded yet, OR
        - Distance moved > 100 meters, OR
        - Force refresh requested by user.
        """
        curr = st.session_state.geo_location_state
        lat = curr.get("latitude")
        lon = curr.get("longitude")

        if lat is None or lon is None:
            return

        last_lat = curr.get("last_geocoded_lat")
        last_lon = curr.get("last_geocoded_lon")

        # Check distance if previously geocoded
        distance = calculate_distance_m(last_lat, last_lon, lat, lon) if last_lat is not None else 999999.0

        if force or last_lat is None or distance > 100.0:
            print(f"[LocationManager] Performing reverse geocoding for coordinates ({lat}, {lon})...")
            geo_data = fetch_reverse_geocoding(lat, lon)
            if geo_data:
                smart_name = format_smart_location_label(geo_data)
                curr.update({
                    "source_location_name": smart_name or f"Location ({lat:.4f}, {lon:.4f})",
                    "full_address_data": geo_data,
                    "last_geocoded_lat": lat,
                    "last_geocoded_lon": lon
                })
            else:
                if not curr.get("source_location_name"):
                    curr.update({
                        "source_location_name": f"Coordinates Available ({lat:.4f}, {lon:.4f})",
                        "full_address_data": None
                    })

    def update_from_browser_result(self, result: dict):
        if not result or not isinstance(result, dict):
            return

        lat = result.get("latitude")
        lon = result.get("longitude")
        acc = result.get("accuracy")
        status = result.get("status")

        if lat is not None and lon is not None:
            st.session_state.geo_location_state.update({
                "status": "GRANTED",
                "latitude": float(lat),
                "longitude": float(lon),
                "accuracy": float(acc) if acc is not None else 10.0,
                "source_type": "LIVE_BROWSER_GEOLOCATION",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "error_message": None
            })
            # Reverse geocode ONCE upon receiving new coordinates
            self.reverse_geocode_current_location()

        elif status in ["DENIED", "UNAVAILABLE", "INSECURE_CONTEXT"]:
            st.session_state.geo_location_state.update({
                "status": status,
                "latitude": None,
                "longitude": None,
                "accuracy": None,
                "error_message": result.get("error_message", "Location permission denied or unavailable."),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })

    def set_configured_fallback(self, lat: float = None, lon: float = None, name: str = None):
        st.session_state.geo_location_state.update({
            "status": "FALLBACK",
            "latitude": float(lat if lat is not None else self.DEFAULT_FALLBACK_LAT),
            "longitude": float(lon if lon is not None else self.DEFAULT_FALLBACK_LON),
            "accuracy": self.DEFAULT_FALLBACK_ACCURACY,
            "source_type": "CONFIGURED_FALLBACK",
            "source_location_name": name or self.DEFAULT_FALLBACK_NAME,
            "full_address_data": {"display_name": name or self.DEFAULT_FALLBACK_NAME},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error_message": None
        })

    def render_location_ui(self):
        """
        Renders the Live Camera Location section using streamlit-geolocation and cached Nominatim reverse geocoding.
        """
        st.markdown("### 📍 LIVE CAMERA LOCATION")
        st.caption("Click the button below to request native browser location permission and acquire real device GPS coordinates.")

        # Official streamlit-geolocation component
        location_data = streamlit_geolocation()

        if location_data and isinstance(location_data, dict):
            self.update_from_browser_result(location_data)

        curr_state = self.state
        status = curr_state.get("status")

        # Display Location Status & Place Details Card
        if status == "GRANTED":
            place_name = curr_state.get("source_location_name") or "Location name temporarily unavailable"
            lat_val = curr_state["latitude"]
            lon_val = curr_state["longitude"]
            acc_val = curr_state["accuracy"]

            st.success(f"📍 **Location Acquired**: `{place_name}`")

            col_btn, col_info = st.columns([1, 4])
            with col_btn:
                if st.button("🔄 Refresh Place Name", key="btn_refresh_geo_name"):
                    self.reverse_geocode_current_location(force=True)
                    st.rerun()

            lc1, lc2, lc3, lc4 = st.columns(4)
            with lc1:
                st.metric("Place", place_name)
            with lc2:
                st.metric("Coordinates", f"{lat_val:.4f}, {lon_val:.4f}")
            with lc3:
                st.metric("Accuracy", f"±{acc_val} m" if acc_val is not None else "N/A")
            with lc4:
                st.metric("Location Status", "LIVE LOCATION")

            # Expandable full address details
            full_data = curr_state.get("full_address_data")
            if full_data:
                with st.expander("🔍 Location Details (Full OpenStreetMap Address)"):
                    st.write(f"**Full Display Name**: {full_data.get('display_name', 'N/A')}")
                    st.json(full_data.get("address", {}))

        elif status == "DENIED":
            st.warning("⚠️ **Location permission denied**. Live detection continues without fake coordinates.")
            if st.button("📍 Use Configured Camera Location", key="btn_use_fallback_denied"):
                self.set_configured_fallback()
                st.rerun()
        elif status == "FALLBACK":
            st.info(f"🟡 **Configured Camera Location Active**: `{curr_state['source_location_name']}` (`Lat: {curr_state['latitude']}`, `Lon: {curr_state['longitude']}`)")
        else:
            st.info("ℹ️ *Click '📍 Get Current Location' above to trigger browser GPS request.*")

        st.caption("ℹ️ *Note: Nominatim reverse geocoding is cached in session state and is NOT requested during individual webcam detection frames.*")

        return curr_state
