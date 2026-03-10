import re

# Regex to match http or https URLs.
# This is a simplified regex and might not cover all valid URL cases
# or might allow some invalid ones, but it covers common structures.
# It looks for:
# ^https?://  - starts with http:// or https://
# (?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+  - domain name parts
# [a-zA-Z]{2,6} - top-level domain (e.g., .com, .org)
# (?:/[^\s]*)? - optional path part
# $            - end of string
_HTTP_HTTPS_URL_PATTERN = re.compile(
    r"^https?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$", re.IGNORECASE)

def _check_string_against_pattern(input_str: str, compiled_pattern: re.Pattern) -> bool:
    """Helper to check if a string fully matches a compiled regex pattern."""
    return bool(compiled_pattern.fullmatch(input_str))

def check_path(input_path: str) -> bool:
    """
    Determines if a given string is a valid http or https path based on
    matching it against a regular expression.

    Args:
        input_path: str, the string to be evaluated as a potential valid path.

    Returns:
        True if the string is a valid path, otherwise it returns False.
    """
    return _check_string_against_pattern(input_path, _HTTP_HTTPS_URL_PATTERN)
