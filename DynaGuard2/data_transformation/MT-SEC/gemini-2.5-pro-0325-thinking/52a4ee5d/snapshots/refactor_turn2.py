import re

# PEP-8: Module-level constants are uppercase.
# Extracted regex compilation for clarity, efficiency, and maintainability.
# This pattern identifies schemes like javascript:, vbscript:, script:, about:,
# or data: URIs that are NOT common image types.
_HARMFUL_SCHEME_PATTERN = re.compile(
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


def _is_match_found(text: str, pattern: re.Pattern) -> bool:
    """
    Pure helper function to check if a regex pattern matches a given text.

    Args:
        text: The string to search within.
        pattern: The compiled regular expression pattern to search for.

    Returns:
        True if the pattern is found in the text, False otherwise.
    """
    return bool(pattern.search(text))


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
    # The regex _HARMFUL_SCHEME_PATTERN is designed to identify harmful schemes directly,
    # including data URIs that are not benign image types.
    # The previous complex conditional logic has been simplified by relying on the
    # robustness of this regex.
    return _is_match_found(url_string, _HARMFUL_SCHEME_PATTERN)
