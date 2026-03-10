import re

# Precompiled regular expressions for performance and clarity.
_SCHEME_RE = re.compile(r'^[\s\x00-\x1f]*([a-zA-Z][a-zA-Z0-9+.\-]*)\s*:', re.IGNORECASE)
_DATA_IMAGE_PREFIX_RE = re.compile(r'^[\s\x00-\x1f]*data\s*:\s*image\/', re.IGNORECASE)

# Set of schemes considered dangerous.
_DANGEROUS_SCHEMES = {"javascript", "vbscript", "livescript", "mocha"}


def check_javascript_scheme(url_str: str) -> bool:
    """
    Determine whether the given string contains a potentially harmful URL scheme.

    Dangerous schemes include:
      - javascript:
      - vbscript:
      - livescript:
      - mocha:
      - data: (except when the data URL is an image, e.g., data:image/png;base64,...)

    Args:
        url_str (str): The string to examine for potentially dangerous URL schemes.

    Returns:
        bool: True if the input string contains a dangerous URL scheme (except safe
              image data URLs); False otherwise.

    Raises:
        re.error: Although no specific exceptions are raised within this function,
                  incorrectly formatted inputs or regex patterns could raise
                  're.error' exceptions.
    """
    s = "" if url_str is None else str(url_str)

    # Identify the leading URL scheme (if any), allowing leading whitespace/control chars.
    m = _SCHEME_RE.match(s)
    if not m:
        return False

    scheme = m.group(1).lower()

    if scheme == "data":
        # Safe only if it's clearly an image data URL.
        if _DATA_IMAGE_PREFIX_RE.match(s):
            return False
        return True

    # Dangerous if in the known set.
    if scheme in _DANGEROUS_SCHEMES:
        return True

    return False
