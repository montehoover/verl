import logging
from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def parse_url(image_url: str):
    """Parse the URL and return the parsed components."""
    return urlparse(image_url)


def validate_url_scheme(parsed_url) -> None:
    """Validate that the URL scheme is allowed."""
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme must be one of {ALLOWED_SCHEMES}")


def validate_url_host(parsed_url) -> None:
    """Validate that the URL host is trusted."""
    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Image host '{parsed_url.hostname}' is not in the trusted list")


def resolve_image_url(image_url: str) -> str:
    """Resolve and validate an image URL from trusted hosts."""
    # Configure logging
    logging.basicConfig(
        filename='url_resolution.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"Attempting to resolve URL: {image_url}")
    
    try:
        parsed_url = parse_url(image_url)
        
        validate_url_scheme(parsed_url)
        validate_url_host(parsed_url)
        
        logger.info(f"Successfully resolved URL: {image_url}")
        return image_url
    except ValueError as e:
        logger.error(f"Failed to resolve URL: {image_url} - Error: {str(e)}")
        raise
