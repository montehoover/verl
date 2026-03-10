from urllib.parse import urlparse

ALLOWED_DOMAINS = {"example.com", "trusted.org", "api.service.net"}

def validate_url_domain(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a predefined set of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is allowed.

    Raises:
        ValueError: If the URL is invalid, malformed, or its domain is not allowed.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")

    try:
        parsed_url = urlparse(url)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid or malformed URL: {url}. Scheme and domain are required.")

    domain = parsed_url.netloc
    # Remove port if present, e.g., "example.com:8080" -> "example.com"
    if ":" in domain:
        domain = domain.split(":", 1)[0]

    if domain in ALLOWED_DOMAINS:
        return True
    else:
        raise ValueError(f"Domain '{domain}' is not allowed for URL: {url}")

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/resource",
        "https://trusted.org/api/data?param=value",
        "ftp://api.service.net",
        "http://example.com:8080/another/path",
        "http://untrusted.com/hack",
        "example.com/only-domain", # Invalid, no scheme
        "http:///path-only", # Invalid, no domain
        "invalid-url-string",
        "http://sub.example.com/path", # This will fail unless "sub.example.com" is in ALLOWED_DOMAINS
    ]

    for test_url in test_urls:
        try:
            if validate_url_domain(test_url):
                print(f"VALID: {test_url}")
        except ValueError as e:
            print(f"INVALID: {test_url} - {e}")

    print("\nTesting with non-string input:")
    try:
        validate_url_domain(123)
    except ValueError as e:
        print(f"INVALID: 123 - {e}")
