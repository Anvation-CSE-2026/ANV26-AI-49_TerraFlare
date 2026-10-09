"""
TerraFlare - Geospatial Visualization Module (Phase 5 & Phase 13 Compatible)
Renders interactive Folium map centered on real camera/device GPS coordinates with accuracy radius and status popups.
"""

import folium
from streamlit_folium import st_folium
import streamlit as st

def render_geospatial_map(
    lat: float,
    lon: float,
    risk_data: dict,
    location_name: str = "Bengaluru Demo Location",
    container=None,
    key_suffix: str = "",
    accuracy_m: float = None
):
    """
    Renders an interactive geospatial map centered at the real source/camera coordinates (lat, lon).
    Displays status-responsive marker showing status, severity, risk score, and accuracy.
    If coordinates are None/invalid, displays 'Location unavailable' notice instead of a fake location.
    """
    target = container if container is not None else st

    with target.container():
        st.markdown("### 🗺️ Geographic Map Integration")

        # Check for valid coordinates
        if lat is None or lon is None:
            st.warning("📍 **Location unavailable**: Real browser GPS coordinates have not been acquired yet. Click **'📍 Get Current Location'** above to authorize browser GPS.")
            return

        status = risk_data.get("status", "NORMAL")
        severity = risk_data.get("severity", "LOW")
        score = risk_data.get("risk_score", 0.0)
        fire_conf = risk_data.get("fire_confidence", 0.0)
        smoke_conf = risk_data.get("smoke_confidence", 0.0)
        source = risk_data.get("source", "Image")
        accuracy = accuracy_m if accuracy_m is not None else risk_data.get("accuracy_m", risk_data.get("location_accuracy_m"))

        # Determine marker icon & color based on status
        if status == "ALERT":
            marker_color = "red"
            marker_icon = "fire"
            icon_prefix = "fa"
            badge_text = "🔴 ALERT"
            popup_header = "🔥 FIRE DETECTED HERE"
            popup_subheader = f"Camera: {location_name}"
        elif status == "WATCH":
            marker_color = "orange"
            marker_icon = "exclamation-triangle"
            icon_prefix = "fa"
            badge_text = "🟡 WATCH"
            popup_header = "🟡 WATCH NOTICE"
            popup_subheader = f"Camera: {location_name}"
        else:
            marker_color = "green"
            marker_icon = "leaf"
            icon_prefix = "fa"
            badge_text = "🟢 NORMAL"
            popup_header = f"📍 {location_name}"
            popup_subheader = "NO ACTIVE FIRE THREAT"

        accuracy_html = f"<strong>Accuracy:</strong> ±{accuracy} m<br/>" if accuracy is not None else ""

        # HTML Popup Content
        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 250px; padding: 4px;">
            <h4 style="margin: 0 0 4px 0; color: #DC2626;">{popup_header}</h4>
            <div style="font-weight: 700; color: #475569; font-size: 0.85rem; margin-bottom: 6px;">
                {popup_subheader}
            </div>
            <div style="background-color: #F8FAFC; border: 1px solid #E2E8F0; padding: 6px 10px; border-radius: 6px; margin-bottom: 8px;">
                <strong>Status:</strong> {badge_text}<br/>
                <strong>Severity:</strong> {severity}<br/>
                <strong>Risk Score:</strong> {score * 100:.1f}%
            </div>
            <div style="font-size: 0.85rem; color: #475569;">
                <strong>Fire Confidence:</strong> {fire_conf * 100:.1f}%<br/>
                <strong>Smoke Confidence:</strong> {smoke_conf * 100:.1f}%<br/>
                <strong>Source:</strong> {source}<br/>
                <strong>Lat:</strong> {lat}<br/>
                <strong>Lon:</strong> {lon}<br/>
                {accuracy_html}
            </div>
        </div>
        """

        acc_str = f" | **Accuracy**: `±{accuracy} m`" if accuracy is not None else ""
        if status == "ALERT":
            st.markdown(f"🔥 **FIRE DETECTED HERE** | **Camera Location**: `{location_name}` | `Lat: {lat}`, `Lon: {lon}`{acc_str}")
        else:
            st.markdown(f"📍 **Current Camera Location**: `{location_name}` | `Lat: {lat}`, `Lon: {lon}`{acc_str}")

        # Create Folium Map centered on actual coordinates
        m = folium.Map(location=[lat, lon], zoom_start=14, tiles="OpenStreetMap")

        # Draw Accuracy Circle if available
        if accuracy is not None and accuracy > 0:
            folium.Circle(
                location=[lat, lon],
                radius=min(accuracy, 2000),  # cap visual radius for readability
                color="#2563EB",
                fill=True,
                fill_opacity=0.15,
                tooltip=f"Device Geolocation Accuracy Radius (±{accuracy}m)"
            ).add_to(m)

        # Add Source / Fire Location Marker (📹 Camera Location)
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=280),
            tooltip=f"{popup_header} - {popup_subheader}",
            icon=folium.Icon(color=marker_color, icon=marker_icon, prefix=icon_prefix)
        ).add_to(m)

        # 🛰️ Render NASA FIRMS Satellite Hotspots & Connector Line (Phase 15)
        sat_ver = risk_data.get("satellite_verification") or {}
        hotspots = sat_ver.get("all_nearby_hotspots", [])
        
        # If single corroborated hotspot exists but all_nearby_hotspots is empty, use main hotspot
        if not hotspots and sat_ver.get("latitude") is not None and sat_ver.get("longitude") is not None:
            hotspots = [{
                "latitude": sat_ver.get("latitude"),
                "longitude": sat_ver.get("longitude"),
                "satellite": sat_ver.get("satellite", "NASA FIRMS"),
                "distance_km": sat_ver.get("distance_km"),
                "acq_date": sat_ver.get("acquisition_date"),
                "acq_time": sat_ver.get("acquisition_time"),
                "confidence": sat_ver.get("confidence"),
                "frp": sat_ver.get("frp")
            }]

        for idx, hp in enumerate(hotspots):
            h_lat = hp.get("latitude")
            h_lon = hp.get("longitude")
            if h_lat is None or h_lon is None:
                continue

            h_sat = hp.get("satellite") or "NASA FIRMS"
            h_dist = hp.get("distance_km")
            h_dist_str = f"{h_dist:.2f} km" if h_dist is not None else "N/A"
            h_acq_date = hp.get("acq_date", "")
            h_acq_time = hp.get("acq_time", "")
            h_conf = hp.get("confidence")
            h_frp = hp.get("frp")

            frp_str = f"{h_frp} MW" if h_frp is not None else "N/A"
            conf_str = str(h_conf).capitalize() if h_conf is not None else "N/A"

            sat_popup_html = f"""
            <div style="font-family: Arial, sans-serif; width: 230px; padding: 4px;">
                <h4 style="margin: 0 0 4px 0; color: #D97706;">🛰️ NASA FIRMS Hotspot</h4>
                <div style="font-size: 0.85rem; color: #334155; line-height: 1.5;">
                    <strong>Satellite:</strong> {h_sat}<br/>
                    <strong>Acquisition:</strong> {h_acq_date} {h_acq_time}<br/>
                    <strong>Distance from camera:</strong> {h_dist_str}<br/>
                    <strong>Confidence:</strong> {conf_str}<br/>
                    <strong>FRP:</strong> {frp_str}<br/>
                    <strong>Lat, Lon:</strong> <code>{h_lat:.4f}, {h_lon:.4f}</code>
                </div>
            </div>
            """

            folium.Marker(
                location=[h_lat, h_lon],
                popup=folium.Popup(sat_popup_html, max_width=260),
                tooltip=f"🛰️ FIRMS Hotspot ({h_sat} — {h_dist_str} from camera)",
                icon=folium.Icon(color="darkpurple", icon="globe", prefix="fa")
            ).add_to(m)

            # Draw connecting polyline between Camera and nearest satellite hotspot if corroborated/ALERT
            if idx == 0 and (status in ["ALERT", "WATCH"] or sat_ver.get("status") == "CORROBORATED"):
                folium.PolyLine(
                    locations=[[lat, lon], [h_lat, h_lon]],
                    color="#F59E0B",
                    weight=3,
                    opacity=0.85,
                    dash_array="6, 8",
                    tooltip=f"📹 Camera ⟷ 🛰️ Hotspot Distance: {h_dist_str}"
                ).add_to(m)

        # Render Map in Streamlit
        map_key = f"map_{lat}_{lon}_{status}_{key_suffix}" if key_suffix else f"map_{lat}_{lon}_{status}"
        st_folium(m, width=None, height=450, use_container_width=True, key=map_key, returned_objects=[])
        st.caption("ℹ️ *Clarification: The map marker 📹 indicates the CAMERA / SOURCE GEOGRAPHIC LOCATION. 🛰️ indicates NASA FIRMS thermal anomaly observations. Bounding boxes represent IMAGE-SPACE object detections.*")

