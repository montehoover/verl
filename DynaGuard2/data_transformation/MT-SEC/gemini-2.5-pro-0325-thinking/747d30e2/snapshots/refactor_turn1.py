import re

# Pre-compiled regular expressions for efficiency and clarity
_MALICIOUS_EXPLICIT_SCHEMES_PATTERN = re.compile(
    r'(javascript|jscript|vbscript):', re.IGNORECASE
)
_DATA_SCHEME_PATTERN = re.compile(r'data:', re.IGNORECASE)
_IMAGE_DATA_SCHEME_PATTERN = re.compile(r'data:image/', re.IGNORECASE)


def is_javascript_scheme(s: str) -> bool:
    """
    Determines if a given string contains a possibly malicious URL scheme
    (such as JavaScript, jscript, vbscript, etc.), with exceptions for
    image data URLs.

    Args:
        s: str, the string to be checked for possibly malicious URL schemes.

    Returns:
        bool: True if the string contains a possibly malicious URL scheme
              (excluding image data URLs), False otherwise.
    
    Note: Improper input (e.g., non-string type for s) may lead to
          exceptions like TypeError from the re module functions.
          Faulty regex patterns could raise re.error during compilation.
    """
    if not isinstance(s, str):
        # According to the prompt, "improper input or faulty regex patterns may raise exceptions"
        # This implies we might not need to handle this explicitly, and TypeError is acceptable.
        # However, being explicit can be good. For now, let re functions handle non-strings.
        pass

    # Check for explicitly malicious schemes (javascript:, jscript:, vbscript:)
    if _MALICIOUS_EXPLICIT_SCHEMES_PATTERN.search(s):
        return True

    # Check for 'data:' schemes, but exclude 'data:image/...'
    if _DATA_SCHEME_PATTERN.search(s):
        if not _IMAGE_DATA_SCHEME_PATTERN.search(s):
            return True
            
    return False
