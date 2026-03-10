from urllib.parse import urlparse
import re
import requests

VALID_SCHEMES = ["http", "https"]
API_PATTERN = r'.*\.api\..*'


def validate_webhook(webhook_link: str) -> str:
    """
    Validate that the provided webhook_link is a properly structured and reachable URL.

    Checks:
    - webhook_link is a non-empty string.
    - Scheme is in VALID_SCHEMES (http or https).
    - Hostname exists and matches API_PATTERN (resembles an API endpoint domain).
    - Attempts to download the URL to confirm reachability.

    Returns:
        The webhook URL (str) if valid and accessible.

    Raises:
        ValueError: If the URL is malformed, domain looks dubious, or unreachable.
    """
    if not isinstance(webhook_link, str):
        raise ValueError("Webhook URL must be a string.")

    link = webhook_link.strip()
    if not link:
        raise ValueError("Webhook URL cannot be empty.")

    parsed = urlparse(link)

    scheme = (parsed.scheme or "").lower()
    if scheme not in VALID_SCHEMES:
        raise ValueError("Invalid webhook URL: scheme must be http or https.")

    hostname = parsed.hostname or ""
    if not hostname:
        raise ValueError("Invalid webhook URL: missing host.")

    if not re.search(API_PATTERN, hostname, flags=re.IGNORECASE):
        raise ValueError("Invalid webhook URL: domain does not resemble a standard API endpoint.")

    try:
        response = requests.get(link, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Webhook URL is unreachable: {exc}") from exc

    return link
