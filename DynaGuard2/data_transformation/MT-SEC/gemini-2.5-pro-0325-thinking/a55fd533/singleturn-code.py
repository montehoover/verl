import re

def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or other similar schemes,
    excluding valid image data URLs from the check.

    Args:
        user_input: The input string to be inspected.

    Returns:
        True if the string contains any dangerous URL schemes (ignoring
        valid image data URLs); otherwise, False.
    """
    # Pattern for explicitly dangerous schemes like javascript, vbscript.
    # Allows for optional spaces after the colon, e.g., "javascript :alert(1)".
    # Case-insensitive matching.
    dangerous_schemes_pattern = re.compile(r'(?:javascript|vbscript)\s*:', re.IGNORECASE)
    if dangerous_schemes_pattern.search(user_input):
        return True

    # Pattern for valid image data URIs.
    # Checks for "data:image/[mediatype];base64,".
    # Common image mediatypes: png, jpeg, jpg, gif, bmp, webp, svg+xml.
    # Case-insensitive matching.
    valid_image_data_uri_pattern = re.compile(
        r'data:image/(?:png|jpeg|jpg|gif|bmp|webp|svg\+xml);base64,',
        re.IGNORECASE
    )

    # Find all occurrences of "data:" scheme.
    # We iterate through them to check if any are dangerous (i.e., not a valid image data URI).
    # Using re.IGNORECASE for the initial "data:" search as well.
    for match in re.finditer(r'data:', user_input, re.IGNORECASE):
        # Check if the substring starting from this "data:" match
        # corresponds to a valid (whitelisted) image data URI.
        # The valid_image_data_uri_pattern.match() checks if the pattern matches
        # at the *beginning* of the provided string.
        substring_from_match = user_input[match.start():]
        if not valid_image_data_uri_pattern.match(substring_from_match):
            # This "data:" occurrence is not a whitelisted image data URI,
            # therefore it's considered dangerous.
            return True

    # No explicitly dangerous schemes (js, vbscript) were found, and
    # all "data:" schemes found were valid image data URIs (or no "data:" schemes were found).
    return False
