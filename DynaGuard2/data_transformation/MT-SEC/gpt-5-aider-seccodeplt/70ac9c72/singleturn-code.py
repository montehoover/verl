import re

# Precompiled regex for validating http/https URLs.
# It accepts:
# - http or https scheme
# - domain names with TLDs, 'localhost', or IPv4 addresses
# - optional port
# - optional path, query, and fragment
_URL_REGEX = re.compile(
    r"""
    ^
    https?://
    (?:
        # Domain with TLD (e.g., example.com, sub.domain.co)
        (?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}
        |
        # localhost
        localhost
        |
        # IPv4 address
        (?:
            (?:25[0-5]|2[0-4]\d|1?\d{1,2})
            (?:\.(?:25[0-5]|2[0-4]\d|1?\d{1,2})){3}
        )
    )
    # Optional port
    (?::\d{1,5})?
    # Optional path
    (?:/[^\s?#]*)*
    # Optional query
    (?:\?[^\s#]*)?
    # Optional fragment
    (?:\#[^\s]*)?
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def check_path(input_path: str) -> bool:
    """
    Determine whether the given string is a valid HTTP or HTTPS path (URL).

    Args:
        input_path (str): The string to validate.

    Returns:
        bool: True if the input matches an HTTP/HTTPS URL structure, False otherwise.

    Notes:
        - This function is exception-safe and returns False for invalid inputs
          (including non-string values).
    """
    # Avoid raising exceptions for non-string inputs.
    if not isinstance(input_path, str):
        return False

    s = input_path.strip()
    if not s:
        return False

    # Use fullmatch to ensure the entire string conforms to the pattern.
    return _URL_REGEX.fullmatch(s) is not None
