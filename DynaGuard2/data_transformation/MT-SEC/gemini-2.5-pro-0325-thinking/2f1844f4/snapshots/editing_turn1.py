import re

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a well-formed URL.

    Args:
        url: The string to validate.

    Returns:
        True if the url is a valid format, False otherwise.
    """
    # Regex to check for a valid URL.
    # This regex is a common pattern and covers many cases, but URL validation can be complex.
    # It checks for scheme (http, https, ftp), domain name, optional port, path, query, and fragment.
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|'  # ...or ipv4
        r'\[?[A-F0-9]*:[A-F0-9:]+\]?)'  # ...or ipv6
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    if not isinstance(url, str):
        return False
        
    try:
        return re.fullmatch(regex, url) is not None
    except Exception:
        # In case of any unexpected error during regex matching,
        # treat as invalid. This fulfills "doesn't raise any exceptions".
        return False

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        "http://www.example.com",
        "https://example.com/path/to/page?query=string#fragment",
        "ftp://user:password@example.com:21/resource",
        "http://localhost:8080",
        "http://127.0.0.1",
        "https://[2001:db8::1]:80/test",
        "www.example.com",  # Invalid (missing scheme)
        "http:// example.com",  # Invalid (space)
        "htp://example.com",  # Invalid (typo in scheme)
        None, # Invalid input type
        123 # Invalid input type
    ]

    for u in urls_to_test:
        is_valid = validate_url(u)
        print(f"URL: '{u}' is {'valid' if is_valid else 'invalid'}")
