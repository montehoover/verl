import re

# List of schemes considered potentially harmful.
# 'data:' is included here; its safety is further determined by checking if it's a known image type.
_HARMFUL_SCHEMES_LIST = [
    "javascript", "vbscript", "data", "file", "about",
    "chrome", "opera", "firefox", "edge", "ms-"
    # Other schemes like 'ms-its', 'ms-settings', etc., could be added if needed.
]

# Regex to match harmful schemes at the beginning of the string.
# Allows for optional leading whitespace.
# Example: "javascript:", "  data:", "vbscript:"
_HARMFUL_SCHEME_PATTERN = re.compile(
    r"^\s*(" + "|".join(re.escape(scheme) for scheme in _HARMFUL_SCHEMES_LIST) + r")\s*:",
    re.IGNORECASE
)

# Regex to specifically identify common image data URLs.
# Example: "data:image/png;base64,...", "data:image/jpeg,...", "  data:image/svg+xml,..."
# It looks for "data:image/" followed by a known image type,
# then optional parameters (like ";base64" or ";charset=utf-8"), and finally a comma.
_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^\s*data:image\/(?:png|jpeg|jpg|gif|webp|svg\+xml)(?:;[^,]*)?,",
    re.IGNORECASE
)


def detect_js_scheme(url_string: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs from the verification.

    Args:
        url_string: The string to examine for possible dangerous URL schemes.

    Returns:
        True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; False otherwise.
        Returns False for non-string inputs.
    """
    if not isinstance(url_string, str):
        return False  # Invalid input type

    # Check if the URL string matches any of the defined harmful schemes.
    harmful_match = _HARMFUL_SCHEME_PATTERN.match(url_string)

    if harmful_match:
        # A potentially harmful scheme was found (e.g., "javascript:", "data:").
        # Now, check if this "data:" scheme is actually a benign image data URL.
        if _IMAGE_DATA_URL_PATTERN.match(url_string):
            # It is a recognized image data URL (e.g., "data:image/png;base64,...").
            # In this context, we consider it safe.
            return False
        else:
            # It matched a harmful scheme (e.g., "javascript:", or "data:" that isn't a known image type)
            # and it's NOT a recognized image data URL. This is considered harmful.
            # This branch handles cases like "javascript:alert(1)" or "data:text/html,..."
            # or the example "data:;base64,..."
            return True
    else:
        # The URL string does not start with any of the schemes listed as harmful.
        # (e.g., "http://", "https://", or relative URLs like "/path/to/page")
        return False
