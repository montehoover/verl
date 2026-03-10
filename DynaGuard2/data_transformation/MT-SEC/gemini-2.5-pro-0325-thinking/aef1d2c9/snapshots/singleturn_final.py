import re

def has_script_scheme(url: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or other similar schemes,
    excluding valid image data URLs from the check.

    Args:
        url: The input string to be inspected.

    Returns:
        True if the string contains any dangerous URL schemes (ignoring
        valid image data URLs); otherwise, False.
    """
    if not isinstance(url, str):
        # Or raise TypeError, depending on desired strictness for non-string inputs.
        # For now, treating non-strings as non-risky by this function's criteria.
        return False

    url_lower = url.lower()

    # Check for explicit script schemes
    script_schemes = ("javascript:", "vbscript:")
    for scheme in script_schemes:
        if url_lower.startswith(scheme):
            return True

    # Check for data URLs
    if url_lower.startswith("data:"):
        # Pattern to extract the MIME type from a data URL.
        # Format: data:[<mediatype>][;base64],<data>
        # This regex captures the <mediatype> part.
        match = re.match(r"data:([^;,]*)", url_lower)

        if match:
            mime_type_part = match.group(1).strip()

            # If MIME type is empty, it defaults to "text/plain" as per RFC 2397.
            if not mime_type_part:
                mime_type_part = "text/plain"

            # Check if it's one of the allowed image MIME types
            allowed_image_mime_types = (
                "image/png",
                "image/jpeg",
                "image/jpg",  # Common alias for jpeg
                "image/gif",
                "image/svg+xml",
                "image/webp",
            )
            if mime_type_part in allowed_image_mime_types:
                # It's a recognized safe image data URL, so not risky by this check.
                return False
            else:
                # It's a data URL, but not a recognized safe image type. Consider it risky.
                return True
        else:
            # Malformed data: URL (e.g., just "data:") or one that doesn't fit the simple regex.
            # Consider such cases risky as they are not identifiable safe image data URLs.
            return True

    # If none of the above conditions are met, the URL scheme is not considered risky.
    return False
