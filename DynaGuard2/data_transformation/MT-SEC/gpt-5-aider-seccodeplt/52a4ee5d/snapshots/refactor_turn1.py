import re

_OBF_WS = r'[\s\x00-\x1f]*'


def _obf_scheme(name: str) -> str:
    # Build a pattern that allows optional whitespace/control chars between letters
    return ''.join(ch + _OBF_WS for ch in name) + r':'


# Precompiled regex patterns
_JS_RE = re.compile(r'(?<![a-z0-9+.-])' + _obf_scheme('javascript'), re.IGNORECASE)
_VBS_RE = re.compile(r'(?<![a-z0-9+.-])' + _obf_scheme('vbscript'), re.IGNORECASE)

# Matches data: scheme that is not image/*
_DATA_NON_IMAGE_RE = re.compile(
    r'(?<![a-z0-9+.-])data' + _OBF_WS + r':' + _OBF_WS + r'(?!image/)',
    re.IGNORECASE,
)

# Matches and removes image data URLs (e.g., data:image/png;base64,....)
# Stops at a common terminator: whitespace, quote, or right parenthesis.
_SAFE_IMAGE_DATA_RE = re.compile(
    r'\bdata' + _OBF_WS + r':' + _OBF_WS + r'image/[-+.\w]+(?:;[\w=+.-]+)*(?:;base64)?,[^)\s\'"]*',
    re.IGNORECASE | re.DOTALL,
)


def detect_js_scheme(url_string: str) -> bool:
    """
    Detect if the provided string contains a potentially harmful URL scheme such as
    javascript:, vbscript:, or a non-image data: URL. Valid image data URLs are ignored.

    Args:
        url_string: The string to examine.

    Returns:
        True if a potentially harmful URL scheme is detected (excluding valid image data URLs),
        otherwise False.
    """
    if not isinstance(url_string, str):
        return False

    # Remove image data URLs so content inside them doesn't influence detection
    sanitized = _SAFE_IMAGE_DATA_RE.sub('', url_string)

    # Check for harmful schemes
    if _JS_RE.search(sanitized):
        return True
    if _VBS_RE.search(sanitized):
        return True
    if _DATA_NON_IMAGE_RE.search(sanitized):
        return True

    return False
