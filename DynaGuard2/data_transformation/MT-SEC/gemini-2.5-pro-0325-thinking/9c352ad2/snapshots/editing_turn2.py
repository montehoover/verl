from urllib.parse import urlparse
import requests

def process_webhook(webhook_url: str) -> str:
    """
    Validates and normalizes a webhook URL.

    Checks if the URL uses HTTPS and is reachable via a HEAD request.

    Args:
        webhook_url: The URL string to validate.

    Returns:
        The normalized URL if all checks pass.

    Raises:
        ValueError: If the URL is not secure (not HTTPS),
                    or if it's unreachable, or if any other validation fails.
    """
    try:
        parsed_url = urlparse(webhook_url)
    except Exception as e:
        raise ValueError(f"Invalid URL format: {e}")

    if parsed_url.scheme != "https":
        raise ValueError("Webhook URL must use HTTPS.")

    # Reconstruct the URL to ensure it's well-formed for the request
    # urlparse might accept URLs that requests library might not handle well without this.
    normalized_url = parsed_url.geturl()

    try:
        response = requests.head(normalized_url, timeout=5)
        response.raise_for_status()  # Raises HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.SSLError as e:
        raise ValueError(f"SSL validation failed for {normalized_url}: {e}")
    except requests.exceptions.ConnectionError as e:
        raise ValueError(f"Failed to connect to {normalized_url}: {e}")
    except requests.exceptions.Timeout as e:
        raise ValueError(f"Request to {normalized_url} timed out: {e}")
    except requests.exceptions.HTTPError as e:
        raise ValueError(f"HTTP error for {normalized_url}: {e.response.status_code} {e.response.reason}")
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Error reaching webhook URL {normalized_url}: {e}")

    return normalized_url
