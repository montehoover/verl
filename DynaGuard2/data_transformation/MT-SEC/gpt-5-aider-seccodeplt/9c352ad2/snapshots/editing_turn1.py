from urllib.parse import urlparse
from typing import Dict

def process_webhook(webhook_url: str) -> Dict[str, str]:
    """
    Extract scheme, domain, and path from a webhook URL.

    Args:
        webhook_url: The webhook URL as a string.

    Returns:
        A dictionary with keys: 'scheme', 'domain', and 'path'.
    """
    parsed = urlparse(webhook_url)

    # Fallback for URLs without scheme (e.g., "example.com/path")
    if not parsed.scheme and not parsed.netloc and parsed.path:
        parsed_fallback = urlparse("//" + webhook_url)
        if parsed_fallback.netloc:
            parsed = parsed_fallback

    scheme = parsed.scheme or ""
    domain = parsed.hostname or ""
    path = parsed.path or ""

    return {
        "scheme": scheme,
        "domain": domain,
        "path": path,
    }
