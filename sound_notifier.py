"""
TERRAFLARE Audible Warning System (Phase 17 - Input-Driven Automatic Fire Alarm Edition)
Provides browser-compatible Web Audio API sound alerts, automatic continuous repeating alarm ONLY when real fire is detected by photo, video, or webcam input,
diagnostic status reporting, and volume/mute controls.
"""

import time
import io
import wave
import math
import struct
import base64
import streamlit as st
import streamlit.components.v1 as components

def generate_synth_wav_base64(tone_type="alert", duration_sec=0.85, sample_rate=22050):
    """
    Generates a lightweight PCM 16-bit mono WAV audio in memory
    and returns a Base64 data URI string.
    """
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        num_samples = int(duration_sec * sample_rate)
        samples = []

        for i in range(num_samples):
            t = i / sample_rate
            if tone_type == "test":
                freq = 587.33 if t < 0.15 else 880.0
                envelope = max(0.0, 1.0 - (t / 0.35))
            else:  # alert
                cycle = int(t / 0.18) % 2
                freq = 880.0 if cycle == 0 else 1200.0
                envelope = max(0.0, 1.0 - (t / 0.85))

            val = int(32767.0 * 0.4 * envelope * math.sin(2.0 * math.pi * freq * t))
            val = max(-32768, min(32767, val))
            samples.append(struct.pack('<h', val))

        wav_file.writeframes(b''.join(samples))

    b64_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    return f"data:audio/wav;base64,{b64_data}"


ALERT_AUDIO_DATA_URI = generate_synth_wav_base64(tone_type="alert", duration_sec=0.85)
TEST_AUDIO_DATA_URI = generate_synth_wav_base64(tone_type="test", duration_sec=0.35)


class SoundAlertManager:
    """
    Manages Web Audio API browser sound alert state, risk transitions, automatic repeating alarm,
    and single Streamlit sidebar UI component for TerraFlare.
    """
    def __init__(self):
        self._init_session_state()

    @staticmethod
    def _init_session_state():
        if "sound_enabled" not in st.session_state:
            st.session_state["sound_enabled"] = True
        if "sound_muted" not in st.session_state:
            st.session_state["sound_muted"] = False
        if "sound_volume" not in st.session_state:
            st.session_state["sound_volume"] = 80  # 80% default
        if "sound_watch_enabled" not in st.session_state:
            st.session_state["sound_watch_enabled"] = False  # Disabled by default
        if "current_risk_status" not in st.session_state:
            st.session_state["current_risk_status"] = "NORMAL"
        if "current_alert_event_id" not in st.session_state:
            st.session_state["current_alert_event_id"] = None
        if "test_event_id" not in st.session_state:
            st.session_state["test_event_id"] = None

    def render_sidebar_controls(self):
        """
        Renders the SINGLE '🔊 Sound Alert Settings' panel and audio controller in the sidebar.
        """
        st.sidebar.header("🔊 Sound Alert Settings")

        enabled = st.sidebar.checkbox(
            "Enable Sound Alerts",
            value=st.session_state.get("sound_enabled", True),
            key="chk_sound_enabled_pb"
        )
        st.session_state["sound_enabled"] = enabled

        muted = st.sidebar.checkbox(
            "Mute Audio",
            value=st.session_state.get("sound_muted", False),
            key="chk_sound_muted_pb"
        )
        st.session_state["sound_muted"] = muted

        volume = st.sidebar.slider(
            "Volume",
            min_value=0,
            max_value=100,
            value=st.session_state.get("sound_volume", 80),
            step=5,
            format="%d%%",
            key="sld_sound_volume_pb"
        )
        st.session_state["sound_volume"] = volume

        watch_enabled = st.sidebar.checkbox(
            "Optional WATCH Audio Notification",
            value=st.session_state.get("sound_watch_enabled", False),
            help="Play low-priority tone when risk is between 0.40 and 0.69 (Disabled by default)",
            key="chk_sound_watch_pb"
        )
        st.session_state["sound_watch_enabled"] = watch_enabled

        if st.sidebar.button("🧪 Test Sound", key="btn_test_sound_sb", use_container_width=True):
            self.trigger_test_sound()

        # Render placeholder container for Web Audio component
        placeholder = st.sidebar.empty()
        st.session_state["sound_audio_placeholder"] = placeholder

        # Render initial Web Audio API Browser Sound Player into placeholder
        self.render_audio_component(placeholder=placeholder)

    def trigger_test_sound(self):
        """
        Triggers a test sound sample directly.
        """
        st.session_state["test_event_id"] = f"test_{int(time.time() * 1000)}"

    def reset_idle_state(self, update_ui=True):
        """
        Resets sound engine to NORMAL / idle state when no active media input is loaded or camera is idle.
        """
        st.session_state["current_risk_status"] = "NORMAL"
        st.session_state["current_alert_event_id"] = None
        if update_ui:
            self.render_audio_component()

    def process_risk_event(self, risk_data, update_ui=True):
        """
        Updates risk status tracking for the sound engine.
        Triggers automatic alarm ONLY when real fire/smoke is detected by photo, video, or webcam input.
        """
        if not risk_data:
            self.reset_idle_state(update_ui=update_ui)
            return

        status = risk_data.get("status", "NORMAL")
        fire_conf = risk_data.get("fire_confidence", 0.0)
        smoke_conf = risk_data.get("smoke_confidence", 0.0)
        ver_res = risk_data.get("verifier_result") or {}
        ver_cls = ver_res.get("result")
        ver_ok = ver_res.get("verified", False)
        ver_supp = ver_res.get("suppression_factor", 1.0)
        is_fire_evt = risk_data.get("is_fire_event", False)

        # Fire is considered REAL and ACTIVE if:
        # 1) Risk Engine Status is "ALERT" (risk_score >= 0.70), OR
        # 2) Fire/Smoke YOLO detection with confidence >= 0.25 (ver_supp >= 0.3), OR
        # 3) CNN Verifier verified FIRE/SMOKE (ver_supp >= 0.4), OR
        # 4) explicitly marked is_fire_event
        is_real_fire = (
            status == "ALERT" or
            is_fire_evt or
            (fire_conf >= 0.25 and ver_supp >= 0.3) or
            (smoke_conf >= 0.25 and ver_supp >= 0.3) or
            (ver_ok and ver_cls in ["FIRE", "SMOKE"] and ver_supp >= 0.4)
        )

        prev_status = st.session_state.get("current_risk_status", "NORMAL")

        if is_real_fire:
            st.session_state["current_risk_status"] = "ALERT"
            if prev_status != "ALERT" or not st.session_state.get("current_alert_event_id"):
                st.session_state["current_alert_event_id"] = f"evt_ALERT_{int(time.time() * 1000000)}"
        else:
            st.session_state["current_risk_status"] = status
            if status != "ALERT":
                st.session_state["current_alert_event_id"] = None

        if update_ui:
            self.render_audio_component()

    def render_audio_component(self, placeholder=None):
        """
        Renders the SINGLE Web Audio API browser sound controller & diagnostic status component in Streamlit.
        """
        if placeholder is None:
            placeholder = st.session_state.get("sound_audio_placeholder")

        sound_enabled = st.session_state.get("sound_enabled", True)
        is_muted = st.session_state.get("sound_muted", False)
        volume_float = float(st.session_state.get("sound_volume", 80)) / 100.0

        current_risk = st.session_state.get("current_risk_status", "NORMAL")
        alert_event_id = st.session_state.get("current_alert_event_id") or ""

        test_event_id = st.session_state.get("test_event_id") or ""
        is_test = bool(test_event_id)

        is_test_js = "true" if is_test else "false"
        is_muted_js = "true" if is_muted else "false"
        sound_enabled_js = "true" if sound_enabled else "false"

        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <meta charset="utf-8"/>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: transparent;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            }}
            .tf-card {{
                background: #121620;
                border: 1px solid #1E293B;
                border-radius: 8px;
                padding: 10px 12px;
                color: #E2E8F0;
                margin-top: 6px;
            }}
            .tf-badge {{
                display: inline-block;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 0.75rem;
                font-weight: 800;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }}
            .status-AUDIO_READY {{ background: #1E3A8A; color: #93C5FD; border: 1px solid #3B82F6; }}
            .status-PLAY_REQUESTED {{ background: #78350F; color: #FDE68A; border: 1px solid #F59E0B; }}
            .status-PLAYING {{ background: #065F46; color: #6EE7B7; border: 1px solid #10B981; animation: pulse 1.0s infinite; }}
            .status-PLAYING_ALERT {{ background: #991B1B; color: #FCA5A5; border: 1px solid #EF4444; animation: pulse 0.8s infinite; }}
            .status-PLAYBACK_BLOCKED {{ background: #7F1D1D; color: #FCA5A5; border: 1px solid #EF4444; }}
            .status-AUDIO_ERROR {{ background: #7F1D1D; color: #FCA5A5; border: 1px solid #EF4444; }}
            .status-MUTED {{ background: #334155; color: #94A3B8; border: 1px solid #64748B; }}
            .status-DISABLED {{ background: #334155; color: #94A3B8; border: 1px solid #475569; }}

            @keyframes pulse {{
                0% {{ opacity: 1.0; }}
                50% {{ opacity: 0.5; }}
                100% {{ opacity: 1.0; }}
            }}

            .tf-msg {{
                font-size: 0.75rem;
                color: #94A3B8;
                line-height: 1.3;
                margin-bottom: 8px;
                min-height: 28px;
            }}
            .tf-btn {{
                width: 100%;
                background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
                color: white;
                border: none;
                border-radius: 6px;
                padding: 7px 10px;
                font-size: 0.8rem;
                font-weight: 700;
                cursor: pointer;
                box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            }}
            .tf-btn:hover {{
                background: linear-gradient(135deg, #DC2626 0%, #B91C1C 100%);
            }}
        </style>
        </head>
        <body>
        <audio id="tf-alert-audio" src="{ALERT_AUDIO_DATA_URI}" loop preload="auto"></audio>
        <audio id="tf-test-audio" src="{TEST_AUDIO_DATA_URI}" preload="auto"></audio>

        <div class="tf-card">
            <div style="font-size: 0.7rem; font-weight: 700; color: #64748B; text-transform: uppercase; margin-bottom: 4px;">
                🔊 Diagnostic Status
            </div>
            <div id="status-badge" class="tf-badge status-{'MUTED' if is_muted else ('DISABLED' if not sound_enabled else ('PLAYING_ALERT' if (current_risk == 'ALERT' and alert_event_id) else 'AUDIO_READY'))}">
                {'MUTED' if is_muted else ('DISABLED' if not sound_enabled else ('PLAYING_ALERT' if (current_risk == 'ALERT' and alert_event_id) else 'AUDIO_READY'))}
            </div>
            <div id="status-msg" class="tf-msg">
                {'Audio muted in settings.' if is_muted else ('Sound alerts disabled.' if not sound_enabled else ('🚨 AUTOMATIC FIRE ALARM PLAYING CONTINUOUSLY' if (current_risk == 'ALERT' and alert_event_id) else 'Web Audio API ready. Silent until input file/camera detects fire.'))}
            </div>
            <button id="tf-play-btn" class="tf-btn" onclick="handleDirectPlayClick()">
                🧪 Test Warning Sound (Direct Click)
            </button>
        </div>

        <script>
        (function() {{
            window.tfSoundState = window.tfSoundState || {{
                audioCtx: null,
                activeAlarmTimer: null,
                activeEventId: null,
                status: "AUDIO_READY"
            }};

            const state = window.tfSoundState;

            const currentRiskStatus = "{current_risk}";
            const alertEventId = "{alert_event_id}";
            const isTestEvent = {is_test_js};
            const testEventId = "{test_event_id}";

            const soundEnabled = {sound_enabled_js};
            const isMuted = {is_muted_js};
            const volume = {volume_float};

            function getAudioContext() {{
                if (!state.audioCtx) {{
                    const AudioContext = window.AudioContext || window.webkitAudioContext;
                    if (AudioContext) {{
                        state.audioCtx = new AudioContext();
                    }}
                }}
                return state.audioCtx;
            }}

            function playFallbackAudio() {{
                try {{
                    const audioElem = document.getElementById("tf-alert-audio");
                    if (audioElem && soundEnabled && !isMuted) {{
                        audioElem.volume = volume;
                        audioElem.play().catch(e => console.warn("HTML5 audio play blocked:", e));
                    }}
                }} catch(e) {{}}
            }}

            function stopFallbackAudio() {{
                try {{
                    const audioElem = document.getElementById("tf-alert-audio");
                    if (audioElem) {{
                        audioElem.pause();
                        audioElem.currentTime = 0;
                    }}
                }} catch(e) {{}}
            }}

            function setDiagnosticStatus(statusKey, msgText) {{
                state.status = statusKey;
                const badge = document.getElementById("status-badge");
                const msgEl = document.getElementById("status-msg");
                if (badge) {{
                    badge.className = "tf-badge status-" + statusKey;
                    badge.innerText = statusKey;
                }}
                if (msgEl) {{
                    msgEl.innerText = msgText;
                }}
            }}

            function playToneOscillator(ctx, toneType, vol, statusCb) {{
                try {{
                    if (ctx.state === "suspended") {{
                        ctx.resume().catch(e => console.warn("ctx resume error:", e));
                    }}
                    const now = ctx.currentTime;
                    const osc = ctx.createOscillator();
                    const gain = ctx.createGain();
                    osc.type = "sawtooth";

                    if (toneType === "test") {{
                        osc.frequency.setValueAtTime(587.33, now);
                        osc.frequency.setValueAtTime(880.0, now + 0.15);

                        gain.gain.setValueAtTime(vol * 0.40, now);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.38);

                        osc.connect(gain);
                        gain.connect(ctx.destination);

                        osc.start(now);
                        osc.stop(now + 0.38);
                        statusCb("PLAYING", "🔊 Continuous Test Alarm Playing (Repeating every 1s)");
                    }} else {{
                        osc.frequency.setValueAtTime(880, now);
                        osc.frequency.setValueAtTime(1200, now + 0.15);
                        osc.frequency.setValueAtTime(880, now + 0.30);
                        osc.frequency.setValueAtTime(1200, now + 0.45);

                        gain.gain.setValueAtTime(vol * 0.50, now);
                        gain.gain.setValueAtTime(vol * 0.50, now + 0.55);
                        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.65);

                        osc.connect(gain);
                        gain.connect(ctx.destination);

                        osc.start(now);
                        osc.stop(now + 0.65);
                        statusCb("PLAYING_ALERT", "🚨 AUTOMATIC CONTINUOUS FIRE ALARM ACTIVE (Repeating every 1s)");
                    }}
                }} catch (err) {{
                    console.error("Oscillator start error:", err);
                    statusCb("AUDIO_ERROR", "Failed to start oscillator: " + err.message);
                }}
            }}

            function playTestTone(ctx, vol) {{
                stopAlarmLoop();
                playToneOscillator(ctx, "test", vol, (statusKey, msgText) => {{
                    setDiagnosticStatus("PLAYING", "🔊 Test Warning Tone Playing");
                    setTimeout(() => {{
                        if (state.status === "PLAYING") {{
                            setDiagnosticStatus("AUDIO_READY", "Web Audio API ready. Silent until input file/camera detects fire.");
                        }}
                    }}, 600);
                }});
                try {{
                    const testElem = document.getElementById("tf-test-audio");
                    if (testElem) {{
                        testElem.volume = vol;
                        testElem.currentTime = 0;
                        testElem.play().catch(e => console.warn("Test audio element play error:", e));
                    }}
                }} catch(e) {{}}
            }}

            function stopAlarmLoop() {{
                if (state.activeAlarmTimer) {{
                    clearInterval(state.activeAlarmTimer);
                    state.activeAlarmTimer = null;
                }}
                state.activeEventId = null;
                stopFallbackAudio();
            }}

            function startAlarmLoop(ctx, evtId, vol, toneType) {{
                if (state.activeAlarmTimer && state.activeEventId === evtId) {{
                    return;
                }}

                stopAlarmLoop();
                state.activeEventId = evtId;

                playToneOscillator(ctx, toneType || "alert", vol, setDiagnosticStatus);
                playFallbackAudio();

                state.activeAlarmTimer = setInterval(() => {{
                    if (!soundEnabled || isMuted) {{
                        stopAlarmLoop();
                        return;
                    }}
                    playToneOscillator(ctx, toneType || "alert", vol, setDiagnosticStatus);
                    playFallbackAudio();
                }}, 1000);
            }}

            function handleUserGesture() {{
                if (!soundEnabled || isMuted) return;
                const ctx = getAudioContext();
                if (ctx && ctx.state === "suspended") {{
                    ctx.resume().then(() => {{
                        if (currentRiskStatus === "ALERT" && alertEventId) {{
                            startAlarmLoop(ctx, alertEventId, volume, "alert");
                        }}
                    }}).catch(e => console.warn("Gesture resume error:", e));
                }} else if (ctx && ctx.state === "running") {{
                    if (currentRiskStatus === "ALERT" && alertEventId && !state.activeAlarmTimer) {{
                        startAlarmLoop(ctx, alertEventId, volume, "alert");
                    }}
                }}
                if (currentRiskStatus === "ALERT" && alertEventId) {{
                    playFallbackAudio();
                }}
            }}

            try {{
                window.addEventListener("click", handleUserGesture, {{ capture: true, passive: true }});
                window.addEventListener("pointerdown", handleUserGesture, {{ capture: true, passive: true }});
                window.addEventListener("keydown", handleUserGesture, {{ capture: true, passive: true }});
                if (window.parent && window.parent.document) {{
                    window.parent.document.addEventListener("click", handleUserGesture, {{ capture: true, passive: true }});
                    window.parent.document.addEventListener("pointerdown", handleUserGesture, {{ capture: true, passive: true }});
                    window.parent.document.addEventListener("keydown", handleUserGesture, {{ capture: true, passive: true }});
                }}
            }} catch(e) {{}}

            window.handleDirectPlayClick = function() {{
                if (!soundEnabled) {{
                    setDiagnosticStatus("DISABLED", "Enable Sound Alerts toggle to play sound.");
                    return;
                }}
                if (isMuted) {{
                    setDiagnosticStatus("MUTED", "Audio is muted in Sound Alert Settings.");
                    return;
                }}

                const ctx = getAudioContext();
                if (!ctx) {{
                    setDiagnosticStatus("AUDIO_ERROR", "Web Audio API is not supported in this browser.");
                    return;
                }}

                ctx.resume().then(() => {{
                    playTestTone(ctx, volume);
                }}).catch(err => {{
                    console.warn("Direct play resume error:", err);
                    playTestTone(ctx, volume);
                }});
            }};

            if (!soundEnabled) {{
                stopAlarmLoop();
                setDiagnosticStatus("DISABLED", "Sound alerts disabled in settings.");
                return;
            }}

            if (isMuted) {{
                stopAlarmLoop();
                setDiagnosticStatus("MUTED", "Audio muted in Sound Alert Settings.");
                return;
            }}

            if (isTestEvent && testEventId) {{
                const ctx = getAudioContext();
                if (ctx) {{
                    if (ctx.state === "suspended") {{
                        ctx.resume().then(() => {{
                            playTestTone(ctx, volume);
                        }}).catch(err => {{
                            playTestTone(ctx, volume);
                        }});
                    }} else {{
                        playTestTone(ctx, volume);
                    }}
                }}
                return;
            }}

            if (currentRiskStatus === "ALERT" && alertEventId) {{
                const ctx = getAudioContext();
                if (ctx) {{
                    if (ctx.state === "suspended") {{
                        ctx.resume().then(() => {{
                            startAlarmLoop(ctx, alertEventId, volume, "alert");
                        }}).catch(err => {{
                            console.warn("Autoplay notice: click anywhere on page to enable audio.");
                            playFallbackAudio();
                        }});
                    }} else {{
                        startAlarmLoop(ctx, alertEventId, volume, "alert");
                    }}
                }} else {{
                    playFallbackAudio();
                }}
            }} else {{
                stopAlarmLoop();
                if (state.status !== "PLAYING" && state.status !== "PLAYING_ALERT") {{
                    setDiagnosticStatus("AUDIO_READY", "Web Audio API ready. Silent until input file/camera detects fire.");
                }}
            }}
        }})();
        </script>
        </body>
        </html>
        """

        target = placeholder if placeholder is not None else st.sidebar
        try:
            with target:
                components.html(html_code, height=135)
        except Exception:
            components.html(html_code, height=135)

        # Clear test event ID after rendering so it won't re-trigger on subsequent reruns
        if is_test:
            st.session_state["test_event_id"] = None
