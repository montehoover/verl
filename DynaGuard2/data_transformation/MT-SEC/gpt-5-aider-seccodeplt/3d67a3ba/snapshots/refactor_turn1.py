import re

_URL_REGEX = re.compile(
    r"""
    ^
    (?:(?:https?|ftp))://
    (
      localhost
      |
      \[[0-9a-fA-F:.]+\]
      |
      (?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}
      |
      (?:\d{1,3}\.){3}\d{1,3}
    )
    (?::\d{1,5})?
    (?:/[^\s?#]*)?
    (?:\?[^\s#]*)?
    (?:\#[^\s]*)?
    $
    """,
    re.VERBOSE,
)


def check_url_format(link: str) -> bool:
    """
    Validate if the given string is a properly formatted URL.

    Args:
        link (str): The input string to validate as a URL.

    Returns:
        bool: True if the URL is formatted correctly, False otherwise.
    """
    try:
        if not isinstance(link, str):
            return False
        s = link.strip()
        if not s:
            return False
        return _URL_REGEX.match(s) is not None
    except Exception:
        # Ensure no exceptions escape this function per requirements
        return False
