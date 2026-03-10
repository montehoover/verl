import re

# Precompiled regular expressions for performance
_SCHEME_RE = re.compile(r'([a-z][a-z0-9+\-.]*)\s*:', re.IGNORECASE)
_DATA_IMAGE_RE = re.compile(
    r'^\s*data\s*:\s*image/[a-z0-9.+-]+(?:;[a-z0-9.+-]+(?:=[a-z0-9.+-]+)?)*(?:;base64)?,',
    re.IGNORECASE,
)

# Known dangerous URL schemes (excluding data:, which is handled specially)
_DANGEROUS_SCHEMES = {
    "javascript",
    "vbscript",
    "file",
    "filesystem",
    "chrome",
    "chrome-extension",
    "moz-extension",
    "ms-browser-extension",
    "resource",
    "jar",
    "about",
    "view-source",
}


def _iter_schemes(url: str):
    """
    Yield tuples of (scheme, start_index) for each scheme-like occurrence in the string.
    """
    for match in _SCHEME_RE.finditer(url):
        yield match.group(1).lower(), match.start()


def _is_dangerous_non_data_scheme(scheme: str) -> bool:
    """
    Return True if the scheme is considered dangerous and is not 'data'.
    """
    return scheme in _DANGEROUS_SCHEMES


def _is_valid_image_data_url(snippet: str) -> bool:
    """
    Return True if the given snippet (starting at 'data:') is a valid image data URL.
    """
    return _DATA_IMAGE_RE.match(snippet) is not None


def _is_risky_data_url(snippet: str) -> bool:
    """
    Return True if the given snippet (starting at 'data:') is considered risky.
    Any data URL that is not a valid image data URL is risky.
    """
    return not _is_valid_image_data_url(snippet)


def has_script_scheme(url: str) -> bool:
    """
    Determine whether the input string contains a risky URL scheme.

    A URL is considered risky if it uses one of a set of dangerous schemes
    (e.g., 'javascript', 'vbscript'), or if it is a data: URL that is not a
    valid image data URL. Valid image data URLs (data:image/...) are ignored.

    Args:
        url: The input string to inspect.

    Returns:
        True if a dangerous scheme is detected (ignoring valid image data URLs),
        otherwise False.
    """
    if not isinstance(url, str) or url == "":
        return False

    for scheme, start in _iter_schemes(url):
        if scheme == "data":
            remainder = url[start:]
            if _is_risky_data_url(remainder):
                return True
            continue

        if _is_dangerous_non_data_scheme(scheme):
            return True

    return False
