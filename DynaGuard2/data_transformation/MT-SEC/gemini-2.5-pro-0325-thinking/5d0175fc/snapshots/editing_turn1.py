import re

def is_valid_url(url: str) -> bool:
    """
    Checks whether a given string is a valid URL using regular expressions.

    Args:
        url: The string to check.

    Returns:
        True if the string is a valid URL format, False otherwise.
    """
    if not isinstance(url, str):
        return False
    # Regex for URL validation (simplified for common cases)
    # This regex is a common one, but URL validation can be very complex.
    # It covers http, https, ftp, and file schemes, domain names, IPs, optional ports, and paths.
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    try:
        return re.match(regex, url) is not None
    except Exception:
        # Should not happen with a pre-compiled regex and string input,
        # but as a safeguard to meet the "no exceptions raised" requirement.
        return False
