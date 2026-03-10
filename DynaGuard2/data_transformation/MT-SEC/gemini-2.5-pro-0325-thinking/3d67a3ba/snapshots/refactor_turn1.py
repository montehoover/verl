import re

def check_url_format(link: str) -> bool:
    """
    Verifies if a given string is a valid URL based on a regular expression pattern.

    Args:
        link: str, the input string to validate as a URL.

    Returns:
        True if the URL is formatted correctly, and False otherwise.
    """
    # A common regex for URL validation. This can be adjusted for more specific needs.
    # This regex checks for http/https/ftp, optional www, domain name, and optional path/query/fragment.
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if re.fullmatch(url_pattern, link):
        return True
    return False
