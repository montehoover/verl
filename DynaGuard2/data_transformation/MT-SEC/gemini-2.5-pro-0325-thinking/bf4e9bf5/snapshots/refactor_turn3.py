import logging
from urllib.parse import urlparse, ParseResult

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _parse_url(image_url: str) -> ParseResult:
    """Parses the given image URL."""
    return urlparse(image_url)


def _validate_scheme(parsed_url: ParseResult) -> None:
    """
    Validates the scheme of the parsed URL.

    Raises:
        ValueError: If the URL scheme is not in ALLOWED_SCHEMES.
    """
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(
            f"Invalid URL scheme: {parsed_url.scheme}. "
            f"Allowed schemes are: {', '.join(ALLOWED_SCHEMES)}"
        )


def _validate_host(parsed_url: ParseResult) -> None:
    """
    Validates the hostname of the parsed URL.

    Raises:
        ValueError: If the URL hostname is not in TRUSTED_IMAGE_HOSTS.
    """
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(
            f"Untrusted image host: {parsed_url.hostname}. "
            f"Trusted hosts are: {', '.join(TRUSTED_IMAGE_HOSTS)}"
        )


def resolve_image_url(image_url: str) -> str:
    """
    Resolves an image URL after verifying that it belongs to a trusted image hosting service.

    Args:
        image_url: The image URL to be resolved.

    Returns:
        The resolved URL if it belongs to a trusted domain.

    Raises:
        ValueError: If the image URL's scheme is not allowed or its domain is not in the trusted list.
    """
    logger = logging.getLogger(__name__ + ".resolve_image_url")
    logger.setLevel(logging.INFO)

    # Configure file handler if not already configured for this logger
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename.endswith("url_resolution.log") for h in logger.handlers):
        # Check specifically for our intended file handler to avoid issues if other handlers exist
        # or if the function is part of a larger app with complex logging.
        # A more robust solution for general purpose library code might involve
        # letting the application configure logging, but this adheres to the prompt.
        # Remove existing handlers of the same type if we want to ensure only one.
        # For simplicity here, we just add if a specific one isn't found.
        
        # A simpler check if we assume this function is the sole configurator for this specific logger:
        if not logger.handlers: 
            log_file_path = "url_resolution.log"
            file_handler = logging.FileHandler(log_file_path)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.info(f"Attempting to resolve URL: {image_url}")

    try:
        parsed_url = _parse_url(image_url)

        _validate_scheme(parsed_url)
        _validate_host(parsed_url)

        logger.info(f"Successfully resolved URL: {image_url}")
        return image_url
    except ValueError as e:
        logger.error(f"Failed to resolve URL '{image_url}': {e}")
        raise
