import re
from typing import Union

def extract_domain(url: str) -> Union[str, bool]:
    """
    Validates a URL and extracts its domain part using regular expressions.

    Args:
        url: The string to check.

    Returns:
        The domain part of the URL if it's valid, False otherwise.
    """
    if not isinstance(url, str):
        return False
    # Regex for URL validation and domain extraction.
    # It covers http, https, ftp schemes, domain names, IPs, optional ports, and paths.
    # The domain part is captured in a named group 'domain'.
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # Scheme: http:// or https:// or ftp:// or ftps://
        r'(?P<domain>(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # Domain name
        r'localhost|'  # localhost
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}))'  # ...or IP address
        r'(?::\d+)?'  # Optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)  # Optional path
    
    try:
        match = re.match(regex, url)
        if match:
            return match.group('domain')
        return False
    except Exception:
        # Should not happen with a pre-compiled regex and string input,
        # but as a safeguard to meet the "no exceptions raised" requirement.
        return False
