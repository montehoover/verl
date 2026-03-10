from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook(webhook_link: str) -> str:
    if not isinstance(webhook_link, str):
        raise ValueError("Webhook URL must be a string.")

    link = webhook_link.strip()
    if not link:
        raise ValueError("Webhook URL cannot be empty.")

    parsed = urlparse(link)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ("http", "https"):
        raise ValueError("Invalid webhook URL: scheme must be http or https.")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Invalid webhook URL: missing host.")

    # Block internal/private addresses by simple prefix/keyword checks
    for prefix in INTERNAL_IP_RANGES:
        if hostname == prefix or hostname.startswith(prefix):
            raise ValueError("Webhook URL points to an internal/private address.")

    # Allow only approved API domains (exact match or subdomain of approved)
    def is_approved(h: str) -> bool:
        for domain in APPROVED_API_DOMAINS:
            d = domain.lower()
            if h == d or h.endswith("." + d):
                return True
        return False

    if not is_approved(hostname):
        raise ValueError("Webhook URL domain is not in the approved list.")

    return link
