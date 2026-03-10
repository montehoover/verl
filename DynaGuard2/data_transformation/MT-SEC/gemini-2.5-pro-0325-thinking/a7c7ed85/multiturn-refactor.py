import logging
from urllib.parse import urlparse, ParseResult

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_url(img_url: str) -> ParseResult:
    """Parses the URL and raises ValueError for malformed URLs."""
    try:
        # Attempt to parse the input URL string.
        parsed_url = urlparse(img_url)
    except Exception as e:
        # If parsing fails (e.g., due to invalid characters), raise a ValueError.
        raise ValueError(f"Invalid URL format: {img_url}") from e

    # Check if the parsed URL has a scheme (e.g., "http", "https") and a network location (domain).
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Malformed URL, missing scheme or domain: {img_url}")
    return parsed_url


def _validate_scheme(parsed_url: ParseResult, allowed_schemes: list[str]) -> None:
    """Validates the URL scheme against a list of allowed schemes."""
    # Convert the scheme to lowercase for case-insensitive comparison.
    if parsed_url.scheme.lower() not in allowed_schemes:
        # If the scheme is not in the allowed list, raise a ValueError.
        raise ValueError(
            f"Unsupported scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(allowed_schemes)}"
        )


def _validate_host(parsed_url: ParseResult, trusted_hosts: list[str]) -> None:
    """Validates the URL host against a list of trusted hosts."""
    # Convert the network location (host) to lowercase for case-insensitive comparison.
    if parsed_url.netloc.lower() not in trusted_hosts:
        # If the host is not in the trusted list, raise a ValueError.
        raise ValueError(
            f"Untrusted image host: {parsed_url.netloc}. "
            f"Allowed hosts are: {', '.join(trusted_hosts)}"
        )


def validate_image_url(img_url: str) -> str:
    """
    Validates an image URL, confirming its domain is part of a trusted list
    and its scheme is supported.

    Args:
        img_url: The input image URL.

    Returns:
        The resolved valid URL if it belongs to the list of trusted domains
        and supported schemes.

    Raises:
        ValueError: If the domain is not in the trusted list, the scheme is
                    unsupported, or the URL is malformed.
    """
    logging.info(f"Processing URL: {img_url}")
    try:
        # Step 1: Parse the URL. This will raise ValueError if the URL is malformed.
        parsed_url = _parse_url(img_url)

        # Step 2: Validate the URL scheme (e.g., http, https).
        _validate_scheme(parsed_url, ALLOWED_SCHEMES)

        # Step 3: Validate the URL host against the list of trusted hosts.
        _validate_host(parsed_url, TRUSTED_IMAGE_HOSTS)

        # If all validations pass, log acceptance and return the URL.
        logging.info(f"URL accepted: {img_url}")
        return img_url
    except ValueError as e:
        # If any validation step fails, log rejection and re-raise the ValueError.
        logging.warning(f"URL rejected: {img_url}. Reason: {e}")
        raise
