import re

_OBF_WS = r'[\s\x00-\x1f]*'


def _build_obfuscated_scheme_pattern(name: str) -> str:
    """Build a pattern that allows optional whitespace/control chars between letters."""
    return ''.join(ch + _OBF_WS for ch in name) + r':'


# Precompiled regex patterns
_JS_SCHEME_RE = re.compile(
    r'(?<![a-z0-9+.-])' + _build_obfuscated_scheme_pattern('javascript'),
    re.IGNORECASE,
)
_VBS_SCHEME_RE = re.compile(
    r'(?<![a-z0-9+.-])' + _build_obfuscated_scheme_pattern('vbscript'),
    re.IGNORECASE,
)

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


def _strip_image_data_urls(text: str) -> str:
    """Remove image data URLs from the provided text."""
    return _SAFE_IMAGE_DATA_RE.sub('', text)


def _has_potentially_harmful_scheme(text: str) -> bool:
    """Check if the text contains potentially harmful URL schemes."""
    if _JS_SCHEME_RE.search(text):
        return True
    if _VBS_SCHEME_RE.search(text):
        return True
    if _DATA_NON_IMAGE_RE.search(text):
        return True
    return False


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

    sanitized_text = _strip_image_data_urls(url_string)
    return _has_potentially_harmful_scheme(sanitized_text)
