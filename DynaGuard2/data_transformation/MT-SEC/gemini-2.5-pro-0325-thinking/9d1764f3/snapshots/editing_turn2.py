from urllib.parse import urlparse
import requests
import re

VALID_SCHEMES = ["http", "https"]
API_PATTERN = r'.*\.api\..*'

def validate_webhook(webhook_link: str) -> str:
    """
    Validates the structure and reachability of a webhook URL.

    Args:
        webhook_link: The URL string to validate.

    Returns:
        The validated URL if it's valid and accessible.

    Raises:
        ValueError: If the URL structure is incorrect, the domain pattern is dubious, or the URL is unreachable.
    """
    parsed_url = urlparse(webhook_link)
    if not parsed_url.scheme or parsed_url.scheme.lower() not in VALID_SCHEMES:
        raise ValueError(f"Invalid URL scheme: '{parsed_url.scheme}'. URL must start with 'http://' or 'https://'.")

    if not parsed_url.netloc:
        raise ValueError("Invalid URL: Missing network location (e.g., domain name).")

    if not re.match(API_PATTERN, parsed_url.netloc):
        raise ValueError(f"Domain '{parsed_url.netloc}' does not appear to be a standard API endpoint.")

    try:
        response = requests.get(webhook_link, timeout=5)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"URL is not reachable or returned an error: {e}")

    return webhook_link
