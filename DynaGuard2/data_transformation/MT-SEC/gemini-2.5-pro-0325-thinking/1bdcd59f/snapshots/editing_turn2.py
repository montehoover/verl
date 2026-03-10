import re
from typing import Union

def extract_domain(url: str) -> Union[str, bool]:
    """
    Validates if the given string is a well-formed URL and extracts its domain.

    Args:
        url: The string to validate.

    Returns:
        The domain part of the URL if valid, False otherwise.
    """
    if not isinstance(url, str):
        return False
    # Regex for URL validation and domain extraction
    # Scheme (http, https, ftp) + :// + domain (captured) + optional path/query/fragment
    # This regex is a common one, but URL validation can be very complex.
    # For more robust validation, a dedicated library might be better.
    regex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
        r'('  # Start of domain capturing group
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r')'  # End of domain capturing group
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)?$', re.IGNORECASE) # Made the path part optional to match domain-only URLs like http://example.com
    
    try:
        match = re.match(regex, url)
        if match:
            return match.group(1)  # Return the captured domain
        return False
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
        "http://example.com", # URL with no path
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
        domain_or_false = extract_domain(u)
        print(f"URL: '{u}' -> Domain: {domain_or_false}")
