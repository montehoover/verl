from typing import Dict
from urllib.parse import urlparse


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
