import re

def validate_url(url: str) -> bool:
    """
    Validates if the given string is a well-formed URL.

    Args:
        url: The string to validate.

    Returns:
        True if the url is a valid format, False otherwise.
    """
    if not isinstance(url, str):
        return False
    # Regex for URL validation (simplified for common cases)
    # Scheme (http, https, ftp) + :// + domain + optional path/query/fragment
    # This regex is a common one, but URL validation can be very complex.
    # For more robust validation, a dedicated library might be better.
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    try:
        return re.match(regex, url) is not None
    except Exception:
        # Should not happen with a pre-compiled regex and string input,
        # but as a safeguard.
        return False

if __name__ == '__main__':
    # Test cases
    urls_to_test = [
        "http://www.example.com",
        "https://example.com/path?query=value#fragment",
        "ftp://user:password@example.com:21/path",
        "http://localhost:8080",
        "http://127.0.0.1",
        "https://example.co.uk",
        "http://example.museum",
        "not_a_url",
        "http://",
        "www.example.com",
        "example.com",
        "http://exa_mple.com", # Invalid hostname character
        "http://-example.com", # Invalid start for domain part
        "http://example-.com", # Invalid end for domain part
        "http://.com",
        "http://localhost:port", # Invalid port
        "http://127.0.0.1:99999", # Invalid port number
        123, # Not a string
        None, # Not a string
    ]

    for u in urls_to_test:
        is_valid = validate_url(u)
        print(f"URL: '{u}' -> Valid: {is_valid}")
