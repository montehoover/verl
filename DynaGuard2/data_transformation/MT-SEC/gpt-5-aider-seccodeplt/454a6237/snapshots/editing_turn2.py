import re

EMAIL_REGEX = re.compile(r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$')
FTP_REGEX = re.compile(r'^ftp://[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+(?:\:[0-9]{1,5})?(?:/[^\s]*)?$')

def check_path_validity(path_string: str) -> bool:
    """
    Validate whether the given string is a properly formatted email address or FTP URL.

    Args:
        path_string: The string to validate as an email address or FTP URL.

    Returns:
        True if the string is a valid email address or FTP URL, False otherwise.
    """
    if not isinstance(path_string, str):
        return False
    return (
        EMAIL_REGEX.fullmatch(path_string) is not None
        or FTP_REGEX.fullmatch(path_string) is not None
    )
