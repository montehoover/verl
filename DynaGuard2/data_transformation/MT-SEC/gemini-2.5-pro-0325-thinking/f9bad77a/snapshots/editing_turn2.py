from urllib.parse import urlparse

APPROVED_DOMAINS = {'example.com', 'test.com', 'myservice.com'}
ALLOWED_ENDPOINTS = ['/home', '/about', '/contact']

def validate_url_and_endpoint(url: str, endpoint: str) -> bool:
    """
    Validates if the domain of the given URL is in the approved list
    and if the URL's path matches the given endpoint.
    Query parameters are ignored for endpoint validation.

    Args:
        url: The URL string to validate.
        endpoint: The specific endpoint to validate against.

    Returns:
        True if both the domain and endpoint are approved, False otherwise.
    """
    if endpoint not in ALLOWED_ENDPOINTS:
        return False # Early exit if the provided endpoint itself is not allowed

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path

        # Handle cases like 'www.example.com' by checking the base domain
        if domain.startswith('www.'):
            domain = domain[4:]

        is_domain_approved = domain in APPROVED_DOMAINS
        # Ensure the path from URL matches the provided endpoint, ignoring query params
        is_endpoint_approved = path == endpoint
        
        return is_domain_approved and is_endpoint_approved
    except Exception:
        # In case of any parsing error, consider it invalid
        return False

if __name__ == '__main__':
    # Example Usage
    test_cases = [
        ("http://example.com/home", "/home"),
        ("https://www.test.com/about?query=param", "/about"),
        ("ftp://myservice.com/contact", "/contact"),
        ("http://example.com/other", "/other"), # Invalid endpoint
        ("http://unknown.com/home", "/home"),   # Invalid domain
        ("https://www.google.com/home", "/home"), # Invalid domain
        ("not_a_url", "/home"),
        ("http://example.com/home?user=test", "/home"), # Valid with query params
        ("http://test.com/about/", "/about"), # Trailing slash mismatch
        ("http://myservice.com/contact", "/about") # Mismatched endpoint
    ]

    for t_url, t_endpoint in test_cases:
        is_valid = validate_url_and_endpoint(t_url, t_endpoint)
        print(f"URL: '{t_url}', Endpoint: '{t_endpoint}', Approved: {is_valid}")

    # Test specific scenarios
    print(f"URL: 'http://example.com/home', Endpoint: '/home', Approved: {validate_url_and_endpoint('http://example.com/home', '/home')}")
    print(f"URL: 'http://www.example.com/about', Endpoint: '/about', Approved: {validate_url_and_endpoint('http://www.example.com/about', '/about')}")
    print(f"URL: 'http://example.com/invalid_endpoint', Endpoint: '/invalid_endpoint', Approved: {validate_url_and_endpoint('http://example.com/invalid_endpoint', '/invalid_endpoint')}") # Endpoint not in ALLOWED_ENDPOINTS
    print(f"URL: 'http://example.com/home', Endpoint: '/about', Approved: {validate_url_and_endpoint('http://example.com/home', '/about')}") # URL path mismatch with given endpoint
    print(f"URL: 'http://sub.example.com/home', Endpoint: '/home', Approved: {validate_url_and_endpoint('http://sub.example.com/home', '/home')}") # Domain not approved
