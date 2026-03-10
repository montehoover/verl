import re

def verify_javascript_protocol(url_input: str) -> bool:
    """
    Analyzes a string to detect potentially harmful URL schemes like 'javascript:'
    or 'vbscript:'. It also flags 'data:' URLs unless they are for images
    (e.g., 'data:image/...').

    Args:
        url_input: The URL string to examine.

    Returns:
        True if a potentially harmful URL scheme is found (excluding valid
        image data URLs), False otherwise.
    
    Raises:
        re.error: Can be raised by 're.compile' or 're.search' if the regex
                  is malformed or an error occurs during matching, though
                  unlikely with the current pattern.
    """
    # This regex pattern is designed to identify harmful URL schemes.
    # It looks for:
    #   1. `javascript` or `vbscript` schemes:
    #      - `\b(javascript|vbscript)\s*:`
    #      - `\b` ensures it matches the whole word (e.g., "javascript" not "myjavascript").
    #      - `\s*:` allows for optional spaces before the colon.
    #   2. `data:` schemes that are NOT for images:
    #      - `\bdata\s*:(?!image\/)`
    #      - `\bdata\s*:` matches "data:" with optional spaces.
    #      - `(?!image\/)` is a negative lookahead. It ensures that "data:"
    #        is NOT immediately followed by "image/", thus filtering out
    #        safe image data URLs.
    # The `re.IGNORECASE` flag makes the search case-insensitive.
    harmful_scheme_pattern = re.compile(
        r"\b(javascript|vbscript)\s*:|"  # Matches 'javascript:' or 'vbscript:'
        r"\bdata\s*:(?!image\/)",        # Matches 'data:' not followed by 'image/'
        re.IGNORECASE
    )

    # Perform the search for the harmful pattern in the input URL.
    match = harmful_scheme_pattern.search(url_input)

    # Guard clause: If no match is found, the URL is considered safe.
    if not match:
        return False
    
    # If a match is found, it means a potentially harmful scheme was detected.
    return True
