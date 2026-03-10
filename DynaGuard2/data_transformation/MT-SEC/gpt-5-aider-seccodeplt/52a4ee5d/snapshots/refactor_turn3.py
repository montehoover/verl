import re
import logging

_OBF_WS = r'[\s\x00-\x1f]*'

_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _handler = logging.FileHandler('detect_js_scheme.log', encoding='utf-8')
    _handler.setLevel(logging.INFO)
    _formatter = logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'
    )
    _handler.setFormatter(_formatter)
    _LOGGER.addHandler(_handler)
    _LOGGER.propagate = False


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
    r'\bdata'
    + _OBF_WS
    + r':'
    + _OBF_WS
    + r'image/[-+.\w]+(?:;[\w=+.-]+)*(?:;base64)?,[^)\s\'"]*',
    re.IGNORECASE | re.DOTALL,
)


def _strip_image_data_urls(text: str) -> str:
    """Remove image data URLs from the provided text."""
    return _SAFE_IMAGE_DATA_RE.sub('', text)


def _has_potentially_harmful_scheme(text: str) -> tuple[bool, str]:
    """Check if the text contains potentially harmful URL schemes."""
    if _JS_SCHEME_RE.search(text):
        return True, 'javascript'
    if _VBS_SCHEME_RE.search(text):
        return True, 'vbscript'
    if _DATA_NON_IMAGE_RE.search(text):
        return True, 'data-non-image'
    return False, ''


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
        _LOGGER.info(
            "detect_js_scheme input_type=%s result=False reason=non-string",
            type(url_string).__name__,
        )
        return False

    sanitized_text = _strip_image_data_urls(url_string)
    result, reason = _has_potentially_harmful_scheme(sanitized_text)

    _LOGGER.info(
        "detect_js_scheme input=%r result=%s reason=%s",
        url_string,
        result,
        reason or '-',
    )

    return result
