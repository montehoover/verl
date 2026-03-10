import re

def has_script_scheme(url: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or other similar schemes,
    excluding valid image data URLs from the check.

    Args:
        url: str, the input string to be inspected for
             potentially malicious URL schemes.

    Returns:
        bool, returning True if the string contains any dangerous
        URL schemes, ignoring valid image data URLs; otherwise, False.
    
    Raises:
        The function doesn't explicitly raise exceptions, but possible
        exceptions like 're.error' could occur due to incorrect
        regex patterns or invalid inputs.
    """
    # Regex to identify valid image data URLs (e.g., data:image/png;base64,...)
    # This pattern allows for various image types (png, jpeg, gif, webp, svg+xml)
    # and optional parameters like charset or base64 encoding.
    image_data_url_pattern = re.compile(
        r"^\s*data:image/(?:png|jpe?g|gif|webp|svg\+xml)(?:;[^,]+)?(?:,|$)", 
        re.IGNORECASE
    )

    if image_data_url_pattern.match(url):
        return False

    # Regex to identify risky URL schemes.
    # This pattern looks for common script-like schemes at the beginning of the URL.
    # It allows for optional whitespace before the scheme.
    risky_scheme_pattern = re.compile(
        r"^\s*(?:javascript|vbscript|script|about|data:(?!image/[a-zA-Z0-9\+\-]+))", 
        re.IGNORECASE
    )

    if risky_scheme_pattern.match(url):
        return True
        
    return False
