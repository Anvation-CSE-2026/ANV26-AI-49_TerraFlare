"""
TerraFlare - Phase 16A: Gmail SMTP Email Alert Notifier
Provides robust Python smtplib email notifications for emergency fire alerts,
non-blocking background dispatch, event cooldown deduplication, Gmail App Password auth,
and precise error categorization (Authentication, Connection, TLS, Recipient).
"""

import os
import ssl
import time
import cv2
import smtplib
import threading
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import base64
import urllib.request
import urllib.error
from typing import Dict, List, Tuple, Optional, Any
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

DEFAULT_SENDER = "pradyumnasenapati79@gmail.com"
DEFAULT_APP_PASS = "pyyq inoi ofah glae"

def get_email_config() -> Dict[str, Any]:
    """
    Retrieves email SMTP configuration from environment variables or project defaults.
    """
    sender = os.getenv("SENDER_EMAIL", DEFAULT_SENDER).strip()
    password = (os.getenv("SENDER_APP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD") or DEFAULT_APP_PASS).strip()
    recipient = os.getenv("DEFAULT_RECIPIENT_EMAIL", DEFAULT_SENDER).strip()
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com").strip()
    
    try:
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        smtp_port = 587

    return {
        "sender_email": sender,
        "sender_password": password,
        "default_recipient": recipient,
        "smtp_server": smtp_server,
        "smtp_port": smtp_port,
        "is_configured": bool(sender and password and "@" in sender)
    }

def is_email_configured() -> bool:
    """Checks if valid sender email and app password credentials are available."""
    config = get_email_config()
    return config["is_configured"]

def should_send_alert_email(
    risk_data: dict,
    last_sent_time: float,
    last_sent_event_id: Optional[str] = None,
    cooldown_seconds: float = 60.0
) -> Tuple[bool, str]:
    """
    Evaluates whether an alert email should be dispatched:
    1. Risk Engine status must be ALERT (risk >= 0.70)
    2. Cooldown policy prevents repeat emails for continuous webcam frames of the same event
    """
    if not risk_data or not isinstance(risk_data, dict):
        return False, "Invalid or missing risk assessment data"

    status = risk_data.get("status")
    if status != "ALERT":
        return False, f"Risk status is '{status}' (alert emails only trigger when status is ALERT)"

    curr_t = time.time()
    event_id = None
    if risk_data.get("simulated_alert"):
        event_id = risk_data["simulated_alert"].get("alert_id")
    elif risk_data.get("detection_event"):
        event_id = risk_data["detection_event"].get("event_id")

    # If same event_id and cooldown active
    if last_sent_event_id and event_id and last_sent_event_id == event_id:
        elapsed = curr_t - last_sent_time
        if elapsed < cooldown_seconds:
            remaining = int(cooldown_seconds - elapsed)
            return False, f"Alert email suppressed (cooldown active: {remaining}s remaining for event {event_id})"

    # Global time cooldown check
    elapsed = curr_t - last_sent_time
    if elapsed < cooldown_seconds:
        remaining = int(cooldown_seconds - elapsed)
        return False, f"Alert email suppressed (global cooldown active: {remaining}s remaining)"

    return True, "Alert email approved for dispatch"

def build_alert_email_content(
    alert_payload: dict,
    recipient_emails: List[str]
) -> Tuple[str, str, str]:
    """
    Constructs Subject, Plain Text Body, and HTML Body for an emergency fire alert email.
    """
    config = get_email_config()
    sender = alert_payload.get("override_sender") or config["sender_email"]
    
    score = alert_payload.get("risk_score", 0.0)
    score_pct = f"{score * 100:.1f}%"
    severity = alert_payload.get("severity", "HIGH")
    status = alert_payload.get("status", "ALERT")
    source = alert_payload.get("source", "Live Camera Feed")
    loc_name = alert_payload.get("source_location") or alert_payload.get("location_name") or "Camera Location"
    lat = alert_payload.get("latitude", "N/A")
    lon = alert_payload.get("longitude", "N/A")
    acc = alert_payload.get("accuracy_m")
    acc_str = f"±{acc} m" if acc is not None else "N/A"
    
    # Detected classes
    fire_conf = alert_payload.get("fire_confidence", 0.0)
    smoke_conf = alert_payload.get("smoke_confidence", 0.0)
    detected_classes = []
    if fire_conf > 0.0:
        detected_classes.append(f"FIRE ({fire_conf*100:.1f}%)")
    if smoke_conf > 0.0:
        detected_classes.append(f"SMOKE ({smoke_conf*100:.1f}%)")
    
    verifier_res = alert_payload.get("verifier_result") or {}
    if verifier_res.get("result") in ["FIRE", "SMOKE"] and not detected_classes:
        detected_classes.append(f"{verifier_res.get('result')} (CNN Verified)")
        
    class_str = ", ".join(detected_classes) if detected_classes else "FIRE / SMOKE"

    # Satellite verification status
    sat_ver = alert_payload.get("satellite_verification") or {}
    sat_status = sat_ver.get("status", "UNAVAILABLE")
    sat_name = sat_ver.get("satellite") or "N/A"
    sat_dist = sat_ver.get("distance_km")
    sat_dist_str = f"{sat_dist:.2f} km" if sat_dist is not None else "N/A"
    sat_msg = sat_ver.get("status_message", "No satellite corroboration data.")

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")

    subject = f"🚨 ALERT: TERRAFLARE Fire Intelligence — FIRE DETECTED NEAR {str(loc_name).upper()}"

    plain_text = f"""
============================================================
🚨 TERRAFLARE — FOREST FIRE EARLY WARNING
============================================================
FIRE DETECTED NEAR: {loc_name}
Timestamp: {timestamp_str}
Status: {status} | Severity: {severity}
Risk Score: {score:.2f} ({score_pct})

DETECTION DETAILS:
------------------------------------------------------------
- Detected Class: {class_str}
- Source Mode: {source}
- Location Name: {loc_name}
- Coordinates: Latitude {lat}, Longitude {lon} (Accuracy: {acc_str})
- Fire Confidence (45%): {fire_conf:.2f}
- Smoke Confidence (20%): {smoke_conf:.2f}

SATELLITE CROSS-VERIFICATION (NASA FIRMS):
------------------------------------------------------------
- Status: {sat_status}
- Satellite Product: {sat_name}
- Nearest Hotspot Distance: {sat_dist_str}
- Details: {sat_msg}

A high-resolution snapshot with YOLO11 bounding boxes is attached.
============================================================
TerraFlare Real-Time AI Wildfire Intelligence System
"""

    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #0F172A; color: #E2E8F0; margin: 0; padding: 20px; }}
            .card {{ background-color: #1E293B; border: 2px solid #EF4444; border-radius: 12px; padding: 24px; max-width: 650px; margin: 0 auto; shadow: 0 10px 15px -3px rgba(0,0,0,0.5); }}
            .header {{ background-color: #450A0A; border-bottom: 2px solid #EF4444; padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
            .header h1 {{ color: #EF4444; font-size: 1.5rem; margin: 0 0 6px 0; font-weight: 800; }}
            .header p {{ color: #FEE2E2; margin: 0; font-size: 0.95rem; }}
            .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; background-color: #EF4444; color: #FFFFFF; }}
            .section {{ margin-bottom: 20px; }}
            .section h3 {{ color: #F8FAFC; border-bottom: 1px solid #334155; padding-bottom: 6px; font-size: 1.1rem; margin-top: 0; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
            td {{ padding: 8px 10px; border-bottom: 1px solid #334155; font-size: 0.95rem; }}
            td.label {{ font-weight: 600; color: #94A3B8; width: 40%; }}
            td.val {{ color: #F8FAFC; font-weight: 700; }}
            .sat-box {{ background-color: #1E1B4B; border: 1px solid #6366F1; border-radius: 8px; padding: 12px 16px; margin-top: 15px; }}
            .footer {{ font-size: 0.8rem; color: #64748B; text-align: center; margin-top: 25px; font-style: italic; }}
        </style>
    </head>
    <body>
        <div class="card">
            <div class="header">
                <h1>🚨 FIRE DETECTED NEAR {str(loc_name).upper()}</h1>
                <p>TERRAFLARE — FOREST FIRE EARLY WARNING</p>
            </div>

            <div class="section">
                <h3>🔥 Emergency Status Overview</h3>
                <table>
                    <tr><td class="label">Status:</td><td class="val"><span class="badge">{status}</span></td></tr>
                    <tr><td class="label">Severity Level:</td><td class="val" style="color: #F87171;">{severity}</td></tr>
                    <tr><td class="label">AI Risk Score:</td><td class="val" style="color: #EF4444;">{score:.2f} ({score_pct})</td></tr>
                    <tr><td class="label">Detected Object(s):</td><td class="val" style="color: #FBBF24;">{class_str}</td></tr>
                    <tr><td class="label">Timestamp:</td><td class="val">{timestamp_str}</td></tr>
                    <tr><td class="label">Source Feed:</td><td class="val">{source}</td></tr>
                </table>
            </div>

            <div class="section">
                <h3>📍 Location & Coordinates</h3>
                <table>
                    <tr><td class="label">Location Name:</td><td class="val">{loc_name}</td></tr>
                    <tr><td class="label">Latitude:</td><td class="val"><code>{lat}</code></td></tr>
                    <tr><td class="label">Longitude:</td><td class="val"><code>{lon}</code></td></tr>
                    <tr><td class="label">Accuracy:</td><td class="val">{acc_str}</td></tr>
                </table>
            </div>

            <div class="sat-box">
                <h4 style="color: #818CF8; margin: 0 0 6px 0;">🛰️ NASA FIRMS Satellite Cross-Verification</h4>
                <div style="font-size: 0.9rem; line-height: 1.5;">
                    <strong>Status:</strong> <span style="color: #FCD34D;">{sat_status}</span><br/>
                    <strong>Satellite Product:</strong> {sat_name}<br/>
                    <strong>Nearest Hotspot Distance:</strong> {sat_dist_str}<br/>
                    <em style="font-size: 0.8rem; color: #A5B4FC;">{sat_msg}</em>
                </div>
            </div>

            <div class="footer">
                Attached: High-resolution camera snapshot with bounding boxes.<br/>
                TerraFlare Wildfire Intelligence System © 2026
            </div>
        </div>
    </body>
    </html>
    """

    return subject, plain_text, html_body

def _connect_and_authenticate_smtp(
    sender_email: str,
    sender_password: str,
    smtp_server: str = "smtp.gmail.com",
    preferred_port: int = 587,
    timeout: float = 15.0
) -> Tuple[Optional[smtplib.SMTP], str, str]:
    """
    Establishes connection and authenticates with Gmail SMTP server.
    Option A: Port 587 STARTTLS (Primary)
    Option B: Port 465 Implicit SSL (Fallback)

    Returns tuple of (server_object, status_code, diagnostic_message).
    Status codes: 'SUCCESS', 'AUTH_FAILED', 'CONN_FAILED', 'TLS_FAILED'
    """
    # Define connection strategies to attempt sequentially
    strategies = []
    if preferred_port == 465:
        strategies = [(465, "SSL"), (587, "STARTTLS")]
    else:
        strategies = [(587, "STARTTLS"), (465, "SSL")]

    last_error_code = "CONN_FAILED"
    last_error_msg = "Could not connect to SMTP server."

    for port, mode in strategies:
        server = None
        stage = "CONNECT"
        try:
            context = ssl.create_default_context()
            if mode == "STARTTLS":
                server = smtplib.SMTP(smtp_server, port, timeout=timeout)
                stage = "EHLO1"
                server.ehlo()
                stage = "STARTTLS"
                server.starttls(context=context)
                stage = "EHLO2"
                server.ehlo()
            else:
                server = smtplib.SMTP_SSL(smtp_server, port, context=context, timeout=timeout)
                stage = "EHLO"
                server.ehlo()

            stage = "LOGIN"
            server.login(sender_email, sender_password)
            return server, "SUCCESS", f"Authenticated via {mode} on port {port}"

        except (smtplib.SMTPAuthenticationError, smtplib.SMTPServerDisconnected) as auth_err:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass
            if stage == "LOGIN":
                return None, "AUTH_FAILED", (
                    "Email delivery is not configured. Administrator authentication is required."
                )
            last_error_code = "CONN_FAILED"
            last_error_msg = f"SMTP disconnect during {stage} on port {port}: {auth_err}"

        except ssl.SSLError as ssl_err:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass
            last_error_code = "TLS_FAILED"
            last_error_msg = f"TLS/SSL handshake error on port {port}: {ssl_err}"

        except (smtplib.SMTPConnectError, TimeoutError, OSError) as conn_err:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass
            last_error_code = "CONN_FAILED"
            last_error_msg = f"Failed to connect to {smtp_server}:{port} — {conn_err}"

        except Exception as generic_err:
            if server:
                try:
                    server.quit()
                except Exception:
                    pass
            last_error_code = "CONN_FAILED"
            last_error_msg = f"SMTP error during {stage} on port {port}: {type(generic_err).__name__} {generic_err}"

    return None, last_error_code, last_error_msg

def _send_via_sendgrid_api(
    sendgrid_key: str,
    sender_email: str,
    valid_recipients: List[str],
    subject: str,
    plain_text: str,
    html_body: str,
    image_bgr: Optional[Any] = None
) -> Dict[str, Any]:
    url = "https://api.sendgrid.com/v3/mail/send"
    
    personalizations = [{
        "to": [{"email": recipient} for recipient in valid_recipients]
    }]
    
    payload = {
        "personalizations": personalizations,
        "from": {
            "email": sender_email,
            "name": "TerraFlare Fire Intelligence"
        },
        "subject": subject,
        "content": [
            {"type": "text/plain", "value": plain_text},
            {"type": "text/html", "value": html_body}
        ]
    }
    
    if image_bgr is not None and hasattr(image_bgr, "size") and image_bgr.size > 0:
        is_success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if is_success:
            b64_data = base64.b64encode(buffer.tobytes()).decode("utf-8")
            payload["attachments"] = [{
                "content": b64_data,
                "filename": "fire_alert_snapshot.jpg",
                "type": "image/jpeg",
                "disposition": "attachment"
            }]
            
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=req_data,
        headers={
            "Authorization": f"Bearer {sendgrid_key}",
            "Content-Type": "application/json"
        },
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=12.0) as resp:
            if resp.status in (200, 202):
                return {
                    "status": "SENT",
                    "error_category": "NONE",
                    "success": True,
                    "message": f"Alert email successfully delivered to {', '.join(valid_recipients)} via SendGrid",
                    "recipients": valid_recipients,
                    "timestamp": time.time()
                }
    except urllib.error.HTTPError as http_err:
        err_body = ""
        try:
            err_body = http_err.read().decode("utf-8")
        except Exception:
            pass
        if "verified Sender Identity" in err_body or "Sender Identity" in err_body or http_err.code == 403:
            return {
                "status": "FAILED",
                "error_category": "AUTH_FAILED",
                "success": False,
                "message": f"SendGrid Verification Required: Please click the verification link sent by SendGrid to {sender_email} to activate sending.",
                "recipients": valid_recipients,
                "timestamp": time.time()
            }
        return {
            "status": "FAILED",
            "error_category": "API_ERROR",
            "success": False,
            "message": f"SendGrid API Error (HTTP {http_err.code}): {err_body[:150]}",
            "recipients": valid_recipients,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "FAILED",
            "error_category": "CONN_FAILED",
            "success": False,
            "message": f"SendGrid dispatch failed: {type(e).__name__} {e}",
            "recipients": valid_recipients,
            "timestamp": time.time()
        }

def send_alert_email(
    image_bgr: Optional[Any],
    recipient_emails: List[str],
    alert_payload: dict,
    is_test: bool = False,
    override_sender: Optional[str] = None,
    override_password: Optional[str] = None
) -> Dict[str, Any]:
    """
    Synchronously dispatches an email via SendGrid API or Python smtplib (Gmail SMTP).
    """
    config = get_email_config()
    sender_email = (override_sender or config["sender_email"]).strip()
    sender_password = (override_password or config["sender_password"]).strip()
    smtp_server = config["smtp_server"]
    smtp_port = config["smtp_port"]
    sendgrid_key = os.getenv("SENDGRID_API_KEY", "").strip()

    # Deduplicate and validate recipients
    valid_recipients = list(set([e.strip() for e in recipient_emails if e and "@" in e.strip()]))
    if not valid_recipients:
        if config["default_recipient"] and "@" in config["default_recipient"]:
            valid_recipients = [config["default_recipient"]]
        else:
            return {
                "status": "RECIPIENT_REJECTED",
                "success": False,
                "message": "No valid recipient email address provided.",
                "timestamp": time.time()
            }

    # If SendGrid API Key is configured, try SendGrid API dispatch first
    if sendgrid_key:
        if is_test:
            subject = "🧪 TEST EMAIL: TERRAFLARE Email Alert Verification"
            plain_text = f"🧪 TERRAFLARE TEST MESSAGE\n\nSender: {sender_email}\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}\nStatus: 🟢 Email dispatch operational via SendGrid."
            html_body = f"<div style='font-family: sans-serif; background-color: #0F172A; color: #E2E8F0; padding: 20px; border-radius: 8px;'><h2 style='color: #10B981;'>🧪 TEST EMAIL: TERRAFLARE Email Alert Verification</h2><p>Provider: SendGrid API</p><p>Sender: {sender_email}</p></div>"
        else:
            subject, plain_text, html_body = build_alert_email_content(alert_payload, valid_recipients)
        return _send_via_sendgrid_api(sendgrid_key, sender_email, valid_recipients, subject, plain_text, html_body, image_bgr)

    if not sender_email or not sender_password or "@" not in sender_email:
        return {
            "status": "NOT CONFIGURED",
            "success": False,
            "message": "Email delivery is not configured. Administrator authentication is required.",
            "timestamp": time.time()
        }

    # Attempt connection and authentication
    server, auth_status, auth_msg = _connect_and_authenticate_smtp(
        sender_email=sender_email,
        sender_password=sender_password,
        smtp_server=smtp_server,
        preferred_port=smtp_port,
        timeout=15.0
    )

    if not server or auth_status != "SUCCESS":
        return {
            "status": "FAILED",
            "error_category": auth_status,
            "success": False,
            "message": auth_msg,
            "recipients": valid_recipients,
            "timestamp": time.time()
        }

    try:
        if is_test:
            subject = "🧪 TEST EMAIL: TERRAFLARE Email Alert Verification"
            plain_text = f"""
============================================================
🧪 TERRAFLARE EMAIL ALERT SYSTEM — TEST MESSAGE
============================================================
This is an automated test email sent from TerraFlare Fire Detection System.

Configuration Status:
- SMTP Server: {smtp_server}
- Sender Address: {sender_email}
- Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}

Result: 🟢 Email dispatch operational. No fire event occurred.
============================================================
"""
            html_body = f"""
            <div style="font-family: sans-serif; background-color: #0F172A; color: #E2E8F0; padding: 20px; border-radius: 8px;">
                <h2 style="color: #10B981;">🧪 TEST EMAIL: TERRAFLARE Email Alert Verification</h2>
                <p>This is a test notification to verify your SMTP configuration.</p>
                <ul>
                    <li><strong>SMTP Server:</strong> {smtp_server}</li>
                    <li><strong>Sender:</strong> {sender_email}</li>
                    <li><strong>Status:</strong> 🟢 Operational</li>
                </ul>
                <p style="color: #94A3B8; font-style: italic;">No actual fire event occurred.</p>
            </div>
            """
        else:
            subject, plain_text, html_body = build_alert_email_content(alert_payload, valid_recipients)

        msg = MIMEMultipart("alternative")
        msg["From"] = sender_email
        msg["To"] = ", ".join(valid_recipients)
        msg["Subject"] = subject

        # Attach plain text and HTML alternatives
        msg.attach(MIMEText(plain_text, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Attach image snapshot if available
        if image_bgr is not None and hasattr(image_bgr, "size") and image_bgr.size > 0:
            is_success, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if is_success:
                image_attachment = MIMEImage(buffer.tobytes(), name="fire_alert_snapshot.jpg")
                image_attachment.add_header("Content-Disposition", "attachment", filename="fire_alert_snapshot.jpg")
                msg.attach(image_attachment)

        # Dispatch email
        server.send_message(msg)

        return {
            "status": "SENT",
            "error_category": "NONE",
            "success": True,
            "message": f"Alert email successfully delivered to {', '.join(valid_recipients)}",
            "recipients": valid_recipients,
            "timestamp": time.time()
        }

    except smtplib.SMTPRecipientsRefused as r_err:
        return {
            "status": "FAILED",
            "error_category": "RECIPIENT_REJECTED",
            "success": False,
            "message": f"Recipient address rejected by Gmail SMTP server: {r_err}",
            "recipients": valid_recipients,
            "timestamp": time.time()
        }
    except Exception as dispatch_err:
        return {
            "status": "FAILED",
            "error_category": "DISPATCH_ERROR",
            "success": False,
            "message": f"SMTP Dispatch Error: {type(dispatch_err).__name__} — {dispatch_err}",
            "recipients": valid_recipients,
            "timestamp": time.time()
        }
    finally:
        if server:
            try:
                server.quit()
            except Exception:
                pass

def send_alert_email_async(
    image_bgr: Optional[Any],
    recipient_emails: List[str],
    alert_payload: dict,
    is_test: bool = False,
    override_sender: Optional[str] = None,
    override_password: Optional[str] = None,
    callback: Optional[Any] = None
) -> threading.Thread:
    """
    Dispatches email notification asynchronously in a background daemon thread
    so the real-time webcam frame loop is NEVER blocked.
    """
    def _async_worker():
        result = send_alert_email(
            image_bgr=image_bgr,
            recipient_emails=recipient_emails,
            alert_payload=alert_payload,
            is_test=is_test,
            override_sender=override_sender,
            override_password=override_password
        )
        if callback and callable(callback):
            try:
                callback(result)
            except Exception as cb_err:
                print(f"[Email Async Worker] Callback Warning: {cb_err}")

    bg_thread = threading.Thread(target=_async_worker, daemon=True)
    bg_thread.start()
    return bg_thread

def send_test_email(
    target_email: Optional[str] = None,
    override_sender: Optional[str] = None,
    override_password: Optional[str] = None
) -> Dict[str, Any]:
    """Sends a test email to verify Gmail SMTP configuration."""
    config = get_email_config()
    recipients = [target_email] if target_email and "@" in target_email else [config["default_recipient"]]
    return send_alert_email(
        image_bgr=None,
        recipient_emails=recipients,
        alert_payload={},
        is_test=True,
        override_sender=override_sender,
        override_password=override_password
    )
