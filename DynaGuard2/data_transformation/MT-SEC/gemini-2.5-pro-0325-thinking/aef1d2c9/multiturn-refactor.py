import re
import logging

# Configure basic logging for demonstration.
# In a real application, logging configuration is typically more centralized.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Module-level compiled regex patterns for efficiency and clarity
# Regex to identify valid image data URLs (e.g., data:image/png;base64,...)
# This pattern allows for various image types (png, jpeg, gif, webp, svg+xml)
# and optional parameters like charset or base64 encoding.
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^\s*data:image/(?:png|jpe?g|gif|webp|svg\+xml)(?:;[^,]+)?(?:,|$)",
    re.IGNORECASE
)

# Regex to identify risky URL schemes.
# This pattern looks for common script-like schemes at the beginning of the URL.
# It allows for optional whitespace before the scheme.
# It also ensures that 'data:' schemes are only considered risky if they are not image data URLs.
_RISKY_SCHEME_PATTERN = re.compile(
    r"^\s*(?:javascript|vbscript|script|about|data:(?!image/[a-zA-Z0-9\+\-]+))",
    re.IGNORECASE
)


def _is_safe_image_data_url(url: str) -> bool:
    """Checks if the URL is a safe (image) data URL using a precompiled regex."""
    return bool(_IMAGE_DATA_URL_PATTERN.match(url))


def _has_risky_scheme_prefix(url: str) -> bool:
    """Checks if the URL starts with a known risky scheme using a precompiled regex."""
    return bool(_RISKY_SCHEME_PATTERN.match(url))


def has_script_scheme(url: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or other similar schemes,
    excluding valid image data URLs from the check.

    Args:
        url: str, the input string to be inspected for
             potentially malicious URL schemes.

    Returns:
        bool, returning True if the string contains any dangerous
        URL schemes, ignoring valid image data URLs; otherwise, False.
    
    Raises:
        The function doesn't explicitly raise exceptions, but possible
        exceptions like 're.error' could occur due to incorrect
        regex patterns or invalid inputs.
    """
    logger.info(f"Checking URL for script scheme: {url}")
    
    is_safe_image = _is_safe_image_data_url(url)
    if is_safe_image:
        logger.info(f"URL identified as a safe image data URL: {url}")
        return False
    
    has_risky_prefix = _has_risky_scheme_prefix(url)
    if has_risky_prefix:
        logger.warning(f"URL identified as having a risky script scheme: {url}")
    else:
        logger.info(f"URL does not have a risky script scheme: {url}")
        
    return has_risky_prefix
