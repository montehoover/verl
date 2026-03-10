import re

_scheme_re = re.compile(r'^\s*([a-zA-Z][a-zA-Z0-9+.\-]*)\s*:', re.IGNORECASE)
_data_image_re = re.compile(r'^\s*data\s*:\s*image\/', re.IGNORECASE)

def _is_harmful_scheme(scheme: str) -> bool:
    """
    Pure function: determines if a scheme is considered harmful.
    """
    return scheme in ('javascript', 'vbscript', 'data')


def _is_safe_scheme(url: str, scheme: str) -> bool:
    """
    Pure function: determines if a scheme should be considered safe based on the full URL.
    Currently allows only data:image/* URLs.
    """
    if scheme == 'data' and _data_image_re.match(url):
        return True
    return False


def analyze_scheme(url: str) -> bool:
    """
    Analyze a string to determine if it contains a potentially harmful URL scheme.

    Returns:
        bool: True if the string uses a dangerous scheme (e.g., javascript:, vbscript:, or data:
              except for valid image data URLs). False otherwise.

    Notes:
        - The function does not intentionally raise specific exceptions.
        - Invalid inputs or malformed regex could lead to re.error exceptions.
    """
    if not isinstance(url, str):
        return False

    match = _scheme_re.match(url)
    if not match:
        return False

    scheme = match.group(1).lower()

    # Safe scheme override (e.g., data:image/*)
    if _is_safe_scheme(url, scheme):
        return False

    # Harmful schemes (javascript:, vbscript:, and non-image data:)
    return _is_harmful_scheme(scheme)
