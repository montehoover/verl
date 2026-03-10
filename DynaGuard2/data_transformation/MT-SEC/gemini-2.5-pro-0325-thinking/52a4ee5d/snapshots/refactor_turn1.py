import re

def detect_js_scheme(url_string: str) -> bool:
    """
    Analyzes a given string to determine if it contains a potentially harmful URL scheme,
    such as javascript, vbscript, or similar, while ignoring image data URLs.

    Args:
        url_string: The string to examine for possible dangerous URL schemes.

    Returns:
        True if the string involves a potentially harmful URL scheme (aside from
        valid image data URLs), False otherwise.
    """
    # Regex to detect schemes like javascript:, vbscript:, etc.
    # It specifically excludes data:image/...
    # (?i) for case-insensitive matching
    # \s* for optional spaces before the colon
    # The pattern looks for scheme names followed by a colon.
    # We are looking for common script-like schemes or data URIs that are not images.
    harmful_scheme_pattern = re.compile(
        r"""
        (?i)                                # Case-insensitive matching
        (?:                                 # Non-capturing group for different schemes
            javascript\s*:                  # javascript:
          | vbscript\s*:                    # vbscript:
          | script\s*:                      # script: (less common as a scheme, but can be indicative)
          | about\s*:                       # about: (e.g. about:blank can be part of exploits)
          | data\s*:                        # data:
            (?!                             # Negative lookahead: not followed by
                image/(?:png|jpeg|gif|webp|svg\+xml) # common image types
            )
        )
        """,
        re.VERBOSE
    )

    if harmful_scheme_pattern.search(url_string):
        # Further check to ensure it's not a benign data:image URL that might have slipped through
        # a more complex regex. The main regex already tries to exclude data:image.
        # This is a simplified check; a more robust one would parse the data URI.
        if "data:image/" in url_string.lower():
             # Check if the match was specifically for "data:" and it's an image
            match = harmful_scheme_pattern.search(url_string)
            if match and match.group(0).lower().startswith("data:"):
                 # If the only match is a data:image then it's not harmful in this context
                if not re.search(r"(?i)(?:javascript|vbscript|script|about)\s*:", url_string):
                    return False # It's a data:image and no other harmful schemes found
        return True
    return False
