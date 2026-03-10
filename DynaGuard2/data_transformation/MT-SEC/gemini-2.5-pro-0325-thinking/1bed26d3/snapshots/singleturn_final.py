import re

# Pre-compiled regular expressions for efficiency and clarity.
# These patterns are used to identify various URL schemes.

# Pattern for common dangerous schemes like javascript: or vbscript:.
# re.IGNORECASE ensures case-insensitivity.
# ^ ensures the pattern matches at the beginning of the string.
_DANGEROUS_SCHEMES_PATTERN = re.compile(
    r"^(?:javascript|vbscript):", 
    re.IGNORECASE
)

# Pattern to identify the beginning of a data: URL.
_DATA_SCHEME_PATTERN = re.compile(
    r"^data:", 
    re.IGNORECASE
)

# Pattern to identify common safe image MIME types within data: URLs.
# This includes png, jpeg/jpg, gif, webp, and svg+xml.
_SAFE_IMAGE_DATA_PATTERN = re.compile(
    r"^data:image/(?:png|jpe?g|gif|webp|svg\+xml)", 
    re.IGNORECASE
)

def check_javascript_scheme(url_str: str) -> bool:
    """
    Determines whether or not a given string contains a potentially harmful URL scheme,
    while excluding image-based data URLs from this evaluation.

    Args:
        url_str: str, the string in question to be examined for potentially
                 dangerous URL schemes.

    Returns:
        bool, returns True if the input string contains a dangerous URL scheme
        (except safe image data URLs); returns False otherwise.

    Raises:
        re.error: Although no specific exceptions are raised directly by this function,
                  incorrectly formatted inputs or internal regex pattern issues could lead
                  to 're.error' exceptions from the 're' module operations.
    """
    # First, check for explicitly dangerous schemes like javascript: or vbscript:.
    if _DANGEROUS_SCHEMES_PATTERN.search(url_str):
        return True

    # Next, handle data: URLs.
    if _DATA_SCHEME_PATTERN.search(url_str):
        # A data: URL is considered potentially harmful unless it's a recognized image type.
        # _SAFE_IMAGE_DATA_PATTERN.search() checks if the URL starts with "data:image/(allowed_type)".
        if not _SAFE_IMAGE_DATA_PATTERN.search(url_str):
            # This is a data: URL, but not one of the allowed image types.
            # Examples: "data:text/html,..." or "data:;base64,...".
            # These are flagged as potentially harmful.
            return True
        else:
            # This is a data:image/... URL (e.g., "data:image/png;base64,...").
            # These are specifically excluded from being flagged as dangerous.
            return False 

    # If the URL does not match any of the dangerous scheme patterns identified above,
    # it is considered safe in the context of this function.
    return False
