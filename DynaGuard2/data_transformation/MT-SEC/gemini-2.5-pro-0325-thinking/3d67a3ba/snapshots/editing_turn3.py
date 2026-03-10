import re

def check_url_format(link):
    """
    Verifies if a given string is a valid URL using a regular expression pattern.

    Args:
        link: The string to check.

    Returns:
        True if the URL is formatted correctly, False otherwise.
    """
    # This regex is a common pattern for URL validation.
    # It checks for a scheme (http, https, ftp), a domain name,
    # and optional path, query parameters, and fragment.
    # It's a comprehensive regex but might not cover all edge cases of valid URLs
    # as per RFC 3986, but it's good for most common cases.
    url_pattern = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https:// or ftp:// or ftps://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    
    return re.match(url_pattern, link) is not None

if __name__ == '__main__':
    # Example Usage
    test_links = [
        "http://www.example.com",
        "https://example.com/path?name=value#fragment",
        "ftp://user:password@example.co.uk:21/resource/file.txt",
        "http://localhost:8080",
        "http://192.168.1.1/page",
        "www.example.com",  # This will be False as it lacks a scheme
        "example.com",      # This will be False
        "http:// example.com", # Invalid due to space
        "htp://example.com",   # Invalid scheme
        "https://.com",        # Invalid domain
        "http://example.c",    # Invalid TLD (too short by this regex's rules)
        "https://example..com" # Invalid domain (double dot)
    ]

    for link_to_test in test_links:
        is_valid = check_url_format(link_to_test)
        print(f"URL: '{link_to_test}' -> Valid: {is_valid}")

    # Specific test cases
    print("\nSpecific Test Cases:")
    print(f"'http://valid.com/path' -> {check_url_format('http://valid.com/path')}") # Expected: True
    print(f"'invalid-url' -> {check_url_format('invalid-url')}") # Expected: False
    print(f"'ftp://my.server/data' -> {check_url_format('ftp://my.server/data')}") # Expected: True
    print(f"'https://www.google.com' -> {check_url_format('https://www.google.com')}") # Expected: True
    print(f"'www.missingprotocol.com' -> {check_url_format('www.missingprotocol.com')}") # Expected: False
