"""
TerraFlare - Phase 15: Satellite Verification UI Component
Renders the 🛰️ SATELLITE CROSS-VERIFICATION dashboard panel, metrics, status banners, controls, and data limitations notice.
"""

import streamlit as st
from typing import Dict, Optional

def render_satellite_panel(
    sat_verification: dict,
    container=None,
    on_refresh_callback=None,
    render_expander: bool = True
):
    """
    Renders the Satellite Cross-Verification UI panel below the AI risk assessment results.
    """
    target = container if container is not None else st

    with target.container():
        st.markdown("### 🛰️ SATELLITE CROSS-VERIFICATION")

        if not sat_verification or not isinstance(sat_verification, dict):
            st.warning("ℹ️ **Satellite data unavailable**: Verification payload missing.")
            return

        status = sat_verification.get("status", "UNAVAILABLE")
        source = sat_verification.get("source", "NASA FIRMS")
        sat_name = sat_verification.get("satellite")
        distance_km = sat_verification.get("distance_km")
        acq_date = sat_verification.get("acquisition_date")
        acq_time = sat_verification.get("acquisition_time")
        confidence = sat_verification.get("confidence")
        frp = sat_verification.get("frp")
        radius_km = sat_verification.get("search_radius_km", 5.0)
        time_window_h = sat_verification.get("time_window_hours", 24)
        status_msg = sat_verification.get("status_message", "")

        # Render Status Banner
        if status == "CORROBORATED":
            banner_color = "#10B981"  # Green
            banner_bg = "#064E3B"
            status_icon = "🛰️"
            status_title = "SATELLITE CORROBORATED"
        elif status == "NOT_CORROBORATED":
            banner_color = "#F59E0B"  # Yellow/Amber
            banner_bg = "#451A03"
            status_icon = "⚠️"
            status_title = "SATELLITE NOT CORROBORATED"
        elif status == "REQUESTING":
            banner_color = "#6366F1"  # Indigo/Purple
            banner_bg = "#1E1B4B"
            status_icon = "⏳"
            status_title = "SATELLITE VERIFICATION IN PROGRESS..."
        else:  # UNAVAILABLE
            banner_color = "#3B82F6"  # Blue
            banner_bg = "#1E3A8A"
            status_icon = "ℹ️"
            status_title = "SATELLITE DATA UNAVAILABLE"

        st.markdown(
            f'''
            <div style="
                background-color: {banner_bg};
                border: 2px solid {banner_color};
                border-radius: 8px;
                padding: 12px 18px;
                margin-bottom: 12px;
                color: #FFFFFF;
                display: flex;
                align-items: center;
                justify-content: space-between;
                flex-wrap: wrap;
                gap: 10px;
            ">
                <div style="font-size: 1.15rem; font-weight: 700; color: {banner_color};">
                    {status_icon} Status: {status_title}
                </div>
                <div style="font-size: 0.9rem; opacity: 0.9;">
                    Source: <strong>{source}</strong> | Search Radius: <strong>{radius_km} km</strong> | Window: <strong>{time_window_h}h</strong>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        if status_msg:
            st.info(f"ℹ️ {status_msg}")

        # Metrics Columns
        m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
        
        with m_col1:
            st.metric("Data Source", source)
        with m_col2:
            st.metric("Satellite Product", sat_name if sat_name else "N/A")
        with m_col3:
            dist_str = f"{distance_km:.2f} km" if distance_km is not None else "N/A"
            st.metric("Nearest Hotspot", dist_str)
        with m_col4:
            acq_str = f"{acq_date} {acq_time}" if acq_date and acq_time else ("N/A")
            st.metric("Acquisition Time", acq_str)
        with m_col5:
            conf_str = str(confidence).capitalize() if confidence is not None else "N/A"
            frp_str = f"{frp} MW" if frp is not None else ""
            val_display = f"{conf_str} {frp_str}".strip() if (confidence or frp) else "N/A"
            st.metric("Confidence / FRP", val_display)

        # Expandable Data Limitations Notice (Section 14)
        if render_expander:
            with st.expander("ℹ️ Satellite Data Limitations & Disclaimers"):
                st.caption(
                    "Satellite cross-verification uses recent NASA FIRMS active-fire observations. "
                    "Satellite observations may not detect every fire due to cloud cover, fire size, observation timing, spatial resolution, and data latency. "
                    "Satellite cross-verification provides an independent corroboration signal and does not replace live ground camera detection."
                )
