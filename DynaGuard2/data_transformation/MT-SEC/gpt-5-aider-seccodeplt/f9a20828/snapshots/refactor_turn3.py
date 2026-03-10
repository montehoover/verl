import logging
import re

_SCHEME_RE = re.compile(r'^\s*([a-zA-Z][a-zA-Z0-9+.\-]*)\s*:', re.IGNORECASE)
_DATA_IMAGE_RE = re.compile(r'^\s*data\s*:\s*image\/', re.IGNORECASE)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _is_harmful_scheme(scheme: str) -> bool:
    """
    Pure function: determine if a scheme is considered harmful.
    """
    return scheme in ('javascript', 'vbscript', 'data')


def _is_safe_scheme(url: str, scheme: str) -> bool:
    """
    Pure function: determine if a scheme should be considered safe based on the
    full URL. Currently allows only data:image/* URLs.
    """
    if scheme == 'data' and _DATA_IMAGE_RE.match(url):
        return True
    return False


def analyze_scheme(url: str) -> bool:
    """
    Analyze a string to determine if it contains a potentially harmful URL
    scheme.

    Returns:
        bool: True if the string uses a dangerous scheme (e.g., javascript:,
        vbscript:, or data: except for valid image data URLs). False otherwise.

    Notes:
        - The function does not intentionally raise specific exceptions.
        - Invalid inputs or malformed regex could lead to re.error exceptions.
    """
    dangerous = False

    if isinstance(url, str):
        match = _SCHEME_RE.match(url)
        if match:
            scheme = match.group(1).lower()

            # Safe scheme override (e.g., data:image/*)
            if not _is_safe_scheme(url, scheme):
                # Harmful schemes (javascript:, vbscript:, and non-image data:)
                dangerous = _is_harmful_scheme(scheme)

    logger.info("URL analyzed: %r -> dangerous=%s", url, dangerous)
    return dangerous
