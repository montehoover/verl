from typing import Dict
from urllib.parse import urlparse

INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def parse_webhook_url(url: str) -> Dict[str, str]:
    """
    Parse a webhook URL and return its basic components.

    Args:
        url: The webhook URL to parse.

    Returns:
        A dictionary containing:
            - scheme: The URL scheme (e.g., 'https')
            - hostname: The hostname (e.g., 'example.com')
            - path: The path portion of the URL (e.g., '/webhook')
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)

    return {
        "scheme": parsed.scheme,
        "hostname": parsed.hostname or "",
        "path": parsed.path or "/",
    }


def detect_internal_url(url: str) -> bool:
    """
    Determine if a given URL points to an internal address.

    Considers the following patterns:
    - 'localhost'
    - '127.0.0.1'
    - IPs starting with '10.'
    - IPs starting with '192.168.'

    Args:
        url: The URL to check.

    Returns:
        True if the URL is considered internal, False otherwise.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return False

    for pattern in INTERNAL_IP_RANGES:
        if pattern.endswith("."):
            if hostname.startswith(pattern):
                return True
        else:
            if hostname == pattern:
                return True

    return False
