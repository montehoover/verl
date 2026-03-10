from urllib.parse import urlparse
import logging

# Configure logging with a clear format showing timestamp, level, and message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# List of trusted image hosting providers that we allow
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
# Only allow secure HTTP protocols
ALLOWED_SCHEMES = ["http", "https"]

def is_scheme_allowed(scheme: str) -> bool:
    """Check if the URL scheme is in our allowed list."""
    return scheme in ALLOWED_SCHEMES

def is_host_trusted(hostname: str) -> bool:
    """Check if the hostname belongs to our trusted providers."""
    return hostname in TRUSTED_IMAGE_HOSTS

def validate_scheme(scheme: str) -> None:
    """
    Validate that the URL uses an allowed protocol scheme.
    Raises ValueError if the scheme is not supported.
    """
    if not is_scheme_allowed(scheme):
        raise ValueError(f"Unsupported scheme: {scheme}. Only {ALLOWED_SCHEMES} are allowed.")

def validate_host(hostname: str) -> None:
    """
    Validate that the hostname is from a trusted image provider.
    Raises ValueError if the host is not in our trusted list.
    """
    if not is_host_trusted(hostname):
        raise ValueError(f"Untrusted domain: {hostname}. Only trusted hosts are allowed.")

def validate_image_url(img_url: str) -> str:
    """
    Validate an image URL by checking its scheme and hostname.
    
    Args:
        img_url: The image URL to validate
        
    Returns:
        The original URL if validation passes
        
    Raises:
        ValueError: If the URL scheme or hostname is not allowed
    """
    # Log the incoming URL validation request
    logger.info(f"Validating URL: {img_url}")
    
    # Parse the URL into its components (scheme, hostname, path, etc.)
    parsed_url = urlparse(img_url)
    
    try:
        # First, validate that the URL uses an allowed protocol (http/https)
        validate_scheme(parsed_url.scheme)
        logger.debug(f"Scheme '{parsed_url.scheme}' is allowed")
        
        # Next, validate that the hostname is from a trusted provider
        validate_host(parsed_url.hostname)
        logger.debug(f"Host '{parsed_url.hostname}' is trusted")
        
        # If all validations pass, log success and return the URL
        logger.info(f"URL accepted: {img_url} (scheme: {parsed_url.scheme}, host: {parsed_url.hostname})")
        return img_url
        
    except ValueError as e:
        # Log the rejection with specific reason
        logger.warning(f"URL rejected: {img_url} - Reason: {str(e)}")
        raise
