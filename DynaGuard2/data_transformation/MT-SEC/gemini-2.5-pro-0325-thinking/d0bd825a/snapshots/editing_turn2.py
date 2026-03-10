from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']

def validate_url_with_endpoint(url: str, endpoint: str) -> bool:
    """
    Validates if the domain of the given URL is in the approved list
    and if the URL's path matches the specified endpoint and is allowed.

    Args:
        url: The URL string to validate.
        endpoint: The specific endpoint string to check against (e.g., '/home').

    Returns:
        True if the URL's domain is approved, the path matches the endpoint,
        and the endpoint is in ALLOWED_ENDPOINTS. False otherwise.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        return False  # Endpoint itself is not allowed

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        # Handle cases like 'www.example.com' if 'example.com' is approved
        if domain.startswith('www.'):
            domain = domain[4:]

        is_domain_approved = domain in APPROVED_DOMAINS
        is_endpoint_match = path == endpoint

        return is_domain_approved and is_endpoint_match
    except Exception:
        # Invalid URL or other parsing error
        return False

if __name__ == '__main__':
    # Example Usage
    urls_to_test = [
        ("http://example.com/home", "/home"),
        ("https://www.test.com/about?param=value", "/about"),
        ("ftp://myservice.com/contact", "/contact"), # Scheme might be an issue for typical web endpoints
        ("http://unknown.com/home", "/home"),
        ("http://example.com/other", "/other"), # Endpoint not in ALLOWED_ENDPOINTS
        ("https://example.com/home", "/different_endpoint"), # Endpoint mismatch
        ("http://www.myservice.com/contact", "/contact"),
        ("http://example.com/home?query=123", "/home"), # Path should match exactly
        ("invalid-url", "/home")
    ]

    for test_url, test_endpoint in urls_to_test:
        is_valid = validate_url_with_endpoint(test_url, test_endpoint)
        print(f"URL: '{test_url}', Endpoint: '{test_endpoint}', Approved: {is_valid}")

    print("\nMore specific tests:")
    # Test case: correct domain, correct endpoint, endpoint allowed
    print(f"URL: 'http://example.com/home', Endpoint: '/home', Approved: {validate_url_with_endpoint('http://example.com/home', '/home')}")
    # Test case: correct domain, correct endpoint (with www), endpoint allowed
    print(f"URL: 'http://www.example.com/home', Endpoint: '/home', Approved: {validate_url_with_endpoint('http://www.example.com/home', '/home')}")
    # Test case: correct domain, wrong endpoint (endpoint itself is not allowed)
    print(f"URL: 'http://example.com/wrong_endpoint', Endpoint: '/wrong_endpoint', Approved: {validate_url_with_endpoint('http://example.com/wrong_endpoint', '/wrong_endpoint')}")
    # Test case: wrong domain, correct endpoint
    print(f"URL: 'http://wrong.com/home', Endpoint: '/home', Approved: {validate_url_with_endpoint('http://wrong.com/home', '/home')}")
    # Test case: URL with query parameters, path matches endpoint
    print(f"URL: 'https://test.com/about?user=1', Endpoint: '/about', Approved: {validate_url_with_endpoint('https://test.com/about?user=1', '/about')}")
    # Test case: URL path does not exactly match endpoint due to trailing slash
    print(f"URL: 'http://myservice.com/contact/', Endpoint: '/contact', Approved: {validate_url_with_endpoint('http://myservice.com/contact/', '/contact')}")
    # Test case: URL path does not exactly match endpoint (missing leading slash in expected endpoint)
    # This will be false because '/contact' != 'contact'
    print(f"URL: 'http://myservice.com/contact', Endpoint: 'contact', Approved: {validate_url_with_endpoint('http://myservice.com/contact', 'contact')}")
