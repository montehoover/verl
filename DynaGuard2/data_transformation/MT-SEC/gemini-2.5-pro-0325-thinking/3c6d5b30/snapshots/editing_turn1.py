from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'another-example.com']

def is_valid_domain(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is allowed, False otherwise.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")
    if not url:
        return False # Or raise ValueError("URL cannot be empty.")

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove port if present, e.g., 'example.com:80' -> 'example.com'
        if ':' in domain:
            domain = domain.split(':')[0]
        return domain in ALLOWED_DOMAINS
    except Exception:
        # Broad exception to catch any parsing errors for malformed URLs
        return False

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.example.com/another/page?query=param",
        "http://sub.example.com/path", # This will be False unless 'sub.example.com' is added
        "ftp://another-example.com/resource",
        "http://example.org/some/path", # Not allowed
        "https://www.google.com", # Not allowed
        "invalid-url", # Invalid
        "", # Empty
        "http://example.com:8080/path" # Check with port
    ]

    for test_url in test_urls:
        is_valid = is_valid_domain(test_url)
        print(f"URL: '{test_url}', Is valid domain? {is_valid}")

    # Test with non-string input
    try:
        is_valid_domain(123)
    except TypeError as e:
        print(f"Error for non-string input: {e}")
