from urllib.parse import urlparse, urlunparse
import requests

ALLOWED_SCHEMES = ["http", "https"]

def verify_image_url(img_url: str) -> str:
    """
    Verifies an image URL by ensuring it uses 'http' or 'https',
    checks if the host is active, and standardizes to 'https'.

    Args:
        img_url: The input image URL string.

    Returns:
        The standardized URL string with 'https' scheme.

    Raises:
        ValueError: If the URL scheme is invalid, the host is unreachable,
                    or the URL returns an HTTP error status.
    """
    parsed_url = urlparse(img_url)

    if not parsed_url.scheme or parsed_url.scheme.lower() not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: '{parsed_url.scheme}'. Must be one of {ALLOWED_SCHEMES}."
        )

    # Ensure the URL to check has a scheme and netloc for the HEAD request
    if not parsed_url.netloc:
        raise ValueError(f"Invalid URL: Missing domain/host name in '{img_url}'.")

    # Use the original img_url for the HEAD request to check reachability
    # with its current scheme (http or https)
    try:
        response = requests.head(img_url, timeout=5, allow_redirects=True)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Unreachable or invalid image URL: {img_url}. Error: {e}")

    # If all checks passed and host is reachable, enforce 'https'
    # The _replace method creates a new namedtuple with specified fields replaced
    https_url_parts = parsed_url._replace(scheme='https')
    return urlunparse(https_url_parts)
