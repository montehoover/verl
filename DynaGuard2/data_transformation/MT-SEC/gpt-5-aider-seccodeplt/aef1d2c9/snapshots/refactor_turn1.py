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

    for match in _SCHEME_RE.finditer(url):
        scheme = match.group(1).lower()

        if scheme in _DANGEROUS_SCHEMES:
            return True

        if scheme == "data":
            # Check if the substring beginning at this scheme is a valid image data URL.
            remainder = url[match.start() :]
            if _DATA_IMAGE_RE.match(remainder):
                # Valid image data URL; ignore and continue searching.
                continue
            # Any other data: URL is considered risky.
            return True

    return False
