"""
TerraFlare - Phase 16: Email Alerts Package
Provides non-blocking SMTP email alerts for fire emergencies with snapshot attachments,
event-driven deduplication, and configuration testing.
"""

from email_alerts.email_notifier import (
    get_email_config,
    is_email_configured,
    send_alert_email,
    send_alert_email_async,
    send_test_email,
    should_send_alert_email
)

__all__ = [
    "get_email_config",
    "is_email_configured",
    "send_alert_email",
    "send_alert_email_async",
    "send_test_email",
    "should_send_alert_email"
]
