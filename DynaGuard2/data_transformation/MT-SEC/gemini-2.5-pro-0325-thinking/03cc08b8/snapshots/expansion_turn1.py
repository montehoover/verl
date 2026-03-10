from urllib.parse import urlparse

ALLOWED_DOMAINS = [
    "example.com",
    "trusted.org",
    "sub.example.com",
]

def validate_url_domain(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a predefined list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is in the ALLOWED_DOMAINS list, False otherwise.

    Raises:
        ValueError: If the URL is invalid or cannot be parsed.
    """
    try:
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("Invalid URL format")
        
        domain = parsed_url.netloc
        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]
            
        return domain in ALLOWED_DOMAINS
    except ValueError as e:
        # Re-raise specific ValueError for invalid URL format
        raise ValueError(f"Invalid URL: {url}. {e}")
    except Exception as e:
        # Catch any other parsing errors and raise as ValueError
        raise ValueError(f"Could not parse URL: {url}. Error: {e}")

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path",
        "https://trusted.org/another/path?query=param",
        "ftp://sub.example.com:8080",
        "http://untrusted.com",
        "example.com/path", # Invalid URL, missing scheme
        "http://example.com:1234/path",
        "https://another.trusted.org",
        "just-a-string",
    ]

    for test_url in test_urls:
        try:
            is_valid = validate_url_domain(test_url)
            print(f"URL: '{test_url}', Domain valid: {is_valid}")
        except ValueError as e:
            print(f"URL: '{test_url}', Error: {e}")

    # Test with a URL that might cause other parsing issues
    try:
        validate_url_domain("http://[::1]:80/path") # IPv6
        print(f"URL: 'http://[::1]:80/path', Domain valid: {validate_url_domain('http://[::1]:80/path')}")
    except ValueError as e:
        print(f"URL: 'http://[::1]:80/path', Error: {e}")
    
    # Test with a domain that is allowed but has a port
    try:
        url_with_port = "https://example.com:443/secure"
        is_valid = validate_url_domain(url_with_port)
        print(f"URL: '{url_with_port}', Domain valid: {is_valid}")
    except ValueError as e:
        print(f"URL: '{url_with_port}', Error: {e}")
