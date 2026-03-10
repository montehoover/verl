from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}

def validate_url_domain(url: str) -> bool:
    """
    Validates if the domain of the given URL is in the approved list.

    Args:
        url: The URL string to validate.

    Returns:
        True if the domain is approved, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Handle cases like 'www.example.com' by checking the base domain
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        # In case of any parsing error, consider it invalid
        return False

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.test.com/another/page?query=param",
        "ftp://myservice.com",
        "http://unknown.com",
        "https://www.google.com",
        "not_a_url"
    ]

    for t_url in test_urls:
        is_valid = validate_url_domain(t_url)
        print(f"URL: '{t_url}', Approved: {is_valid}")

    # Test with a subdomain of an approved domain (should be False by current logic)
    print(f"URL: 'http://sub.example.com', Approved: {validate_url_domain('http://sub.example.com')}")
    # Test with an approved domain without www
    print(f"URL: 'http://example.com', Approved: {validate_url_domain('http://example.com')}")
    # Test with an approved domain with www
    print(f"URL: 'http://www.example.com', Approved: {validate_url_domain('http://www.example.com')}")
