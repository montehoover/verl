from urllib.parse import urlparse

ALLOWED_DOMAINS = ['example.com', 'trusted.org', 'sub.example.com']

def validate_return_url(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is allowed.

    Raises:
        ValueError: If the URL's domain is not in the allowed list or if the URL is malformed.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    if domain is None:
        raise ValueError(f"Could not extract domain from URL: {url}")

    if domain in ALLOWED_DOMAINS:
        return True
    else:
        # Check for subdomains if a wildcard isn't explicitly used
        # For example, if 'example.com' is allowed, 'www.example.com' should also be allowed.
        # This logic can be adjusted based on specific requirements.
        parts = domain.split('.')
        for i in range(len(parts) - 1):
            parent_domain = '.'.join(parts[i:])
            if parent_domain in ALLOWED_DOMAINS:
                return True
        
        raise ValueError(f"Domain '{domain}' is not allowed. URL: {url}")

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://trusted.org/another/page?query=param",
        "http://sub.example.com/some/resource",
        "ftp://example.com/file",
        "http://www.example.com/test", # Should be allowed if example.com is
        "https://another.trusted.org/path", # Should be allowed if trusted.org is
        "http://untrusted.com/malicious",
        "http://example.net/path",
        "just_a_string",
        "",
        "http://localhost:8000/path" # Example of a domain not in the list
    ]

    for t_url in test_urls:
        try:
            if validate_return_url(t_url):
                print(f"URL '{t_url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{t_url}': {e}")

    # Example with a new allowed domain
    ALLOWED_DOMAINS.append("localhost")
    print("\n--- After adding 'localhost' to ALLOWED_DOMAINS ---")
    try:
        if validate_return_url("http://localhost:8000/path"):
            print(f"URL 'http://localhost:8000/path' is valid.")
    except ValueError as e:
        print(f"Validation failed for URL 'http://localhost:8000/path': {e}")
