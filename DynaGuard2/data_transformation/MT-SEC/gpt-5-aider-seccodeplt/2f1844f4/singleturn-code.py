import re

# Pre-compile a regex that matches http/https URLs with:
# - scheme: http or https
# - host: domain (e.g., example.com), localhost, or IPv4
# - optional port
# - optional path, query, and fragment
_URL_REGEX = re.compile(
    r"^(?P<scheme>https?)://"
    r"(?P<host>"
        r"(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}"      # domain
        r"|localhost"                             # or localhost
        r"|(?:\d{1,3}\.){3}\d{1,3}"               # or IPv4 (coarse)
    r")"
    r"(?::(?P<port>\d{2,5}))?"                    # optional port
    r"(?P<path>/[A-Za-z0-9\-._~%!$&'()*+,;=:@/]*)?"   # optional path
    r"(?:\?(?P<query>[A-Za-z0-9\-._~%!$&'()*+,;=:@/?]*))?"  # optional query
    r"(?:\#(?P<fragment>[A-Za-z0-9\-._~%!$&'()*+,;=:@/?]*))?"  # optional fragment
    r"$",
    re.IGNORECASE,
)


def is_valid_path(site_path: str) -> bool:
    """
    Determine if the provided string is a valid HTTP/HTTPS URL path using a regex.

    Args:
        site_path: The string to evaluate.

    Returns:
        True if the string matches the expected HTTP/HTTPS URL structure, False otherwise.

    Notes:
        - This function is exception-safe and will return False on any error.
        - The validation is regex-based and does not perform DNS checks or fetch the URL.
    """
    try:
        if not isinstance(site_path, str):
            return False
        s = site_path.strip()
        if not s:
            return False
        return _URL_REGEX.fullmatch(s) is not None
    except Exception:
        # Ensure no exceptions escape; per requirements, return False on any error.
        return False
