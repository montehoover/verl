import re
import logging

# Configure logger
logger = logging.getLogger(__name__)


def _is_harmful_direct_scheme(scheme: str) -> bool:
    """Checks if the scheme is directly harmful (e.g., javascript, vbscript)."""
    harmful_schemes = {'javascript', 'vbscript'}
    return scheme in harmful_schemes

def _is_safe_data_uri(processed_url: str) -> bool:
    """Checks if a 'data' URI is for a safe image type."""
    # Comparison is case-insensitive for the 'data:image/' part.
    return processed_url.lower().startswith('data:image/')

def analyze_scheme(url: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs from
    the verification.

    Args:
        url: str, the string to examine for possible dangerous URL schemes.

    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.

    Raises:
        The function doesn't raise specific exceptions. However, invalid inputs
        or malformed regex could lead to 're.error' exceptions from the 're' module.
    """
    processed_url = url.strip()

    if not processed_url:
        return False

    # Regex to extract the scheme from the URL.
    # A scheme consists of a letter followed by any combination of letters, digits, "+", ".", or "-".
    # It is followed by a colon. This conforms to RFC 3986.
    scheme_match = re.match(r"([a-zA-Z][a-zA-Z0-9+.-]*):", processed_url)

    if not scheme_match:
        # No valid scheme found at the beginning of the string.
        return False

    scheme = scheme_match.group(1).lower()

    if _is_harmful_direct_scheme(scheme):
        return True

    is_harmful = False
    if _is_harmful_direct_scheme(scheme):
        is_harmful = True
    elif scheme == 'data':
        # A 'data' URI is harmful if it's not a safe image data URI.
        is_harmful = not _is_safe_data_uri(processed_url)
    # else: is_harmful remains False (default)

    if is_harmful:
        logger.warning(f"Processed URL: '{url}'. Identified as HARMFUL.")
    else:
        logger.info(f"Processed URL: '{url}'. Identified as SAFE.")

    return is_harmful
