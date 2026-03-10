import re

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs.

    Args:
        url_input: str, the string to examine for possible dangerous URL schemes.

    Returns:
        bool, returns True if the string involves a potentially harmful URL scheme,
        aside from valid image data URLs; returns False otherwise.
    
    Raises:
        The function doesn't raise specific exceptions. However, invalid inputs
        or malformed regex could lead to 're.error' exceptions from the 're' module.
    """
    # Regex to find "javascript:", "vbscript:", 
    # or "data:" not followed by "image/" (case-insensitive).
    # \b ensures word boundaries for scheme names (e.g., "javascript" not "myjavascript").
    # \s* allows for optional spaces after the scheme name and before the colon.
    # (?!image\/) is a negative lookahead asserting that "data:" is not followed by "image/".
    harmful_scheme_pattern = re.compile(
        r"\b(javascript|vbscript)\s*:|\bdata\s*:(?!image\/)",
        re.IGNORECASE
    )

    # Guard clause: if no harmful scheme is found, the URL is considered safe.
    if not harmful_scheme_pattern.search(url_input):
        return False
    
    # If the guard clause didn't exit, a harmful scheme was detected.
    return True
