import re

# Regex to identify potentially harmful URL schemes (javascript, vbscript, or generic data)
# Allows leading whitespace. Case-insensitive.
HARMFUL_PROTOCOLS_PATTERN = re.compile(
    r"^\s*(?:javascript|vbscript|data):", 
    re.IGNORECASE
)

# Regex to identify common safe image data URLs.
# Allows leading whitespace. Case-insensitive.
# Includes common image types like png, jpeg, jpg, gif, webp, svg+xml.
SAFE_IMAGE_DATA_URL_PATTERN = re.compile(
    r"^\s*data:image/(?:png|jpeg|jpg|gif|webp|svg\+xml);base64",
    re.IGNORECASE
)

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or a generic data: URL, while ignoring recognized
    image data URLs from the verification.

    Args:
        url_input: The string to examine for possible dangerous URL schemes.

    Returns:
        True if the string involves a potentially harmful URL scheme (and is not
        a recognized safe image data URL); False otherwise.
    
    Raises:
        The function itself doesn't raise specific exceptions for typical string inputs.
        However, underlying 're' module functions might raise 're.error' for
        malformed regex patterns (not applicable here as patterns are fixed) or
        if non-string input bypasses type hinting in some contexts (though
        re.search handles None by raising TypeError).
    """
    if not isinstance(url_input, str):
        # Or raise TypeError, depending on desired strictness for non-string inputs.
        # For now, treating non-strings as not matching.
        return False

    # Check if the URL starts with a potentially harmful protocol
    is_potentially_harmful_scheme = bool(HARMFUL_PROTOCOLS_PATTERN.search(url_input))

    if not is_potentially_harmful_scheme:
        # If it's not javascript:, vbscript:, or data:, it's not considered harmful by this function.
        return False

    # At this point, the scheme is javascript:, vbscript:, or data:.
    # Now, we need to ensure it's not a safe image data URL.
    # If it's javascript: or vbscript:, it won't match SAFE_IMAGE_DATA_URL_PATTERN.
    # If it's data:, this check determines if it's a safe image type or a generic (potentially harmful) data URL.
    is_safe_image = bool(SAFE_IMAGE_DATA_URL_PATTERN.search(url_input))

    # A URL is considered harmful if it matches a harmful scheme AND is NOT a safe image data URL.
    return not is_safe_image
