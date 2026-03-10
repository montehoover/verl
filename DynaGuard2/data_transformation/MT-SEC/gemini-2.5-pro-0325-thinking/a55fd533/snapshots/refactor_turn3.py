import re
import logging

# Set up a logger for this module
logger = logging.getLogger(__name__)

# Pattern to find any scheme. Catches the scheme name in group 1.
# Example: "http:", "javascript:", "data:"
# \b ensures it's a whole word before the colon for the scheme name.
# The scheme name itself ([a-zA-Z][a-zA-Z0-9+.-]*) is captured.
# We use .lower() on the captured group, so IGNORECASE is not needed here.
SCHEME_PATTERN = re.compile(r"\b([a-zA-Z][a-zA-Z0-9+.-]*):")

# Pattern to identify safe image data URLs.
# Case-insensitive for "data:image/" and the image types.
# This pattern checks the beginning of a data URL string.
SAFE_IMAGE_DATA_URL_PATTERN = re.compile(
    r"data:image/(?:png|jpeg|jpg|gif|webp|svg\+xml)", 
    re.IGNORECASE
)

# Set of known dangerous schemes (lowercase).
# 'data' is handled specially: it's considered dangerous if it's not a safe image data URL.
DANGEROUS_SCHEMES_LOWERCASE = {
    "javascript",
    "vbscript",
    "file",
    # Other schemes like 'jar', 'moz-binding', etc., could be added here if needed.
}


def _extract_schemes(text: str, pattern: re.Pattern):
    """
    Finds all occurrences of a regex pattern in the given text and yields match objects.

    Args:
        text: The string to search within.
        pattern: The compiled regular expression pattern to search for.

    Yields:
        re.Match: A match object for each occurrence of the pattern.
    """
    for match in pattern.finditer(text):
        yield match


def contains_dangerous_scheme(user_input: str) -> bool:
    """
    Determines whether a given string contains a risky URL scheme,
    such as javascript, vbscript, or other similar schemes,
    excluding valid image data URLs from the check.

    Args:
        user_input: The input string to be inspected for potentially malicious URL schemes.

    Returns:
        bool: True if the string contains any dangerous URL schemes,
              ignoring valid image data URLs; otherwise, False.
    
    The function doesn't explicitly raise exceptions for typical string inputs.
    Incorrect regex patterns could lead to 're.error' (compile-time), 
    and non-string inputs could lead to 'TypeError' from regex functions if not handled.
    """
    logger.info(f"Checking input for dangerous schemes: '{user_input[:100]}{'...' if len(user_input) > 100 else ''}'")

    if not isinstance(user_input, str):
        # Based on type hint, user_input is str. If other types are possible,
        # this check would be important. Depending on requirements,
        # could raise TypeError or return False.
        # For this implementation, we proceed assuming valid string input as per type hint.
        # If an error should be raised for non-strings, a TypeError would be appropriate.
        # If it should silently return False for non-strings:
        # return False
        pass # Assuming valid string input

    # Pipeline Stage 1: Extract all potential schemes from the input.
    # Then, iterate over these schemes to check for dangerous ones.
    for match in _extract_schemes(user_input, SCHEME_PATTERN):
        # match.group(1) is the scheme name itself, e.g., "http", "Javascript"
        scheme_name = match.group(1).lower()
        
        if scheme_name in DANGEROUS_SCHEMES_LOWERCASE:
            logger.warning(f"Dangerous scheme '{scheme_name}' found in input.")
            return True
            
        if scheme_name == "data":
            # The scheme is "data". Now check if it's a safe image data URL.
            # We test SAFE_IMAGE_DATA_URL_PATTERN against the part of the
            # user_input string that starts with "data:".
            # user_input[match.start(0):] gives the substring from the start of the current match.
            url_candidate = user_input[match.start(0):]

            if SAFE_IMAGE_DATA_URL_PATTERN.match(url_candidate):
                # It's a 'data:' URL that matches the pattern for safe image types.
                # This is not considered dangerous, so continue checking other parts of user_input.
                continue
            else:
                # It's a 'data:' URL, but not a recognized safe image type (e.g., data:text/html).
                # This is considered dangerous.
                logger.warning(f"Dangerous 'data:' URI found (not a safe image type): '{url_candidate[:100]}{'...' if len(url_candidate) > 100 else ''}'")
                return True
                
    logger.info("No dangerous schemes found in input.")
    return False
