from urllib.parse import urlparse
import requests


def validate_webhook(webhook_link: str) -> str:
    """
    Validate that the provided webhook_link is a properly structured and reachable URL.

    - Ensures webhook_link is a non-empty string.
    - Ensures the URL has a scheme of http or https and a network location.
    - Attempts to download the URL to confirm reachability.

    Returns:
        The normalized webhook URL (str) if valid and accessible.

    Raises:
        ValueError: If the URL is malformed or unreachable.
    """
    if not isinstance(webhook_link, str):
        raise ValueError("Webhook URL must be a string.")

    link = webhook_link.strip()
    if not link:
        raise ValueError("Webhook URL cannot be empty.")

    parsed = urlparse(link)
    if parsed.scheme.lower() not in ("http", "https") or not parsed.netloc:
        raise ValueError("Invalid webhook URL: must include http or https scheme and a valid host.")

    try:
        response = requests.get(link, timeout=10)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise ValueError(f"Webhook URL is unreachable: {exc}") from exc

    return link
