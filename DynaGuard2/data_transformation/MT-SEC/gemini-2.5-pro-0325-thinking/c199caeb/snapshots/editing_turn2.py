from urllib.parse import urlparse, urlencode, urlunparse

APPROVED_DOMAINS = ['example.com', 'trusted.com']

def is_valid_domain(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of approved domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the domain is approved, False otherwise.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove www. if present, as APPROVED_DOMAINS might not have it
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain in APPROVED_DOMAINS
    except Exception:
        # Handle potential parsing errors for malformed URLs
        return False

def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    """
    Constructs a URL from base_url, path, and query_params,
    then validates its domain against APPROVED_DOMAINS.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/api/data").
        query_params: A dictionary of query parameters (e.g., {"id": 123, "type": "user"}).

    Returns:
        The full URL string if the domain is valid.

    Raises:
        ValueError: If the domain of the constructed URL is not in APPROVED_DOMAINS.
    """
    # Ensure path starts with a slash if base_url doesn't end with one and path is not empty
    if not base_url.endswith('/') and path and not path.startswith('/'):
        full_path = '/' + path
    else:
        full_path = path

    # Construct URL without query parameters first to parse and validate domain
    # urlparse requires a scheme, so we ensure base_url has one.
    # If not, we might need to assume 'http' or 'https' or let urlparse handle it.
    # For simplicity, we assume base_url is well-formed enough for urlparse.
    
    parsed_base = urlparse(base_url)
    
    # Reconstruct the base and path part
    # Ensure path is correctly joined, avoiding double slashes if base_url has a path
    # and path argument also starts with a slash.
    
    # If base_url has a path component, and the new path is relative, join them.
    # If the new path is absolute (starts with '/'), it replaces the base_url's path.
    
    # A simpler approach for combining base_url and path:
    # Let urlparse handle the scheme and netloc from base_url.
    # The path component will be the new path.
    # Query parameters are added separately.

    # Construct the URL parts for urlunparse
    scheme = parsed_base.scheme
    netloc = parsed_base.netloc
    
    # Ensure path is correctly formed
    final_path = parsed_base.path.rstrip('/') + '/' + path.lstrip('/') if path else parsed_base.path
    if not final_path.startswith('/'): # Ensure path always starts with / if not empty
        final_path = '/' + final_path
    
    query_string = urlencode(query_params)
    
    # Construct the full URL
    # scheme, netloc, path, params, query, fragment
    constructed_url = urlunparse((scheme, netloc, final_path, '', query_string, ''))

    if not is_valid_domain(constructed_url):
        raise ValueError(f"Domain for URL {constructed_url} is not approved.")
    
    return constructed_url

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.trusted.com/another/page",
        "http://sub.example.com/resource", # This will be False unless sub.example.com is added
        "ftp://untrusted.net/file",
        "http://example.com",
        "https://trusted.com",
        "malformed-url"
    ]

    for test_url in test_urls:
        print(f"URL: {test_url}, Is Valid Domain: {is_valid_domain(test_url)}")

    print("\nTesting with www prefix variations:")
    APPROVED_DOMAINS.append("www.another-trusted.com") # Add with www for testing
    print(f"Is 'http://another-trusted.com' valid? {is_valid_domain('http://another-trusted.com')}") # Should be False
    print(f"Is 'http://www.another-trusted.com' valid? {is_valid_domain('http://www.another-trusted.com')}") # Should be True

    # Test case where approved domain does not have www but URL does
    print(f"Is 'https://www.example.com/path' valid? {is_valid_domain('https://www.example.com/path')}") # Should be True

    # Test case where approved domain has www but URL does not (less common, but good to be aware)
    # Current logic handles this by stripping www from URL, so if 'www.domain.com' is approved,
    # 'domain.com' in URL will be checked against 'www.domain.com' (False)
    # and 'www.domain.com' in URL will be checked against 'www.domain.com' (True after stripping www from URL and comparing to 'www.domain.com')
    # Let's refine APPROVED_DOMAINS to not have 'www.' for consistency or handle it symmetrically.
    # For now, the current logic is: strip www from URL, then check.
    # So, if 'www.explicit-www.com' is in APPROVED_DOMAINS,
    # is_valid_domain('http://explicit-www.com') -> 'explicit-www.com' in APPROVED_DOMAINS -> False
    # is_valid_domain('http://www.explicit-www.com') -> 'explicit-www.com' in APPROVED_DOMAINS -> False
    # This means APPROVED_DOMAINS should store domains without 'www.'
    # Let's adjust the example and logic slightly for clarity.

    # Re-adjusting for clarity: APPROVED_DOMAINS should store base domains.
    # The function will strip 'www.' from the input URL's domain before checking.
    print("\nRefined test for 'www.' handling (APPROVED_DOMAINS should be base domains):")
    # Ensure APPROVED_DOMAINS are base domains for this test
    if 'www.another-trusted.com' in APPROVED_DOMAINS:
        APPROVED_DOMAINS.remove('www.another-trusted.com')
    if 'another-trusted.com' not in APPROVED_DOMAINS:
        APPROVED_DOMAINS.append('another-trusted.com')

    print(f"Current APPROVED_DOMAINS: {APPROVED_DOMAINS}")
    print(f"Is 'http://another-trusted.com' valid? {is_valid_domain('http://another-trusted.com')}")
    print(f"Is 'http://www.another-trusted.com' valid? {is_valid_domain('http://www.another-trusted.com')}")
    print(f"Is 'https://example.com/test' valid? {is_valid_domain('https://example.com/test')}")
    print(f"Is 'https://www.example.com/test' valid? {is_valid_domain('https://www.example.com/test')}")
    print(f"Is 'https://unapproved.com/test' valid? {is_valid_domain('https://unapproved.com/test')}")
    print(f"Is 'https://www.unapproved.com/test' valid? {is_valid_domain('https://www.unapproved.com/test')}")

    print("\nTesting construct_and_validate_url:")
    try:
        valid_url = construct_and_validate_url("http://example.com", "api/resource", {"id": "1", "name": "test"})
        print(f"Constructed valid URL: {valid_url}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        valid_url_with_www = construct_and_validate_url("https://www.trusted.com", "/path", {"key": "value"})
        print(f"Constructed valid URL (www): {valid_url_with_www}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # Test with base_url having a path
        valid_url_base_path = construct_and_validate_url("http://example.com/basepath", "api/resource", {"id": "2"})
        print(f"Constructed valid URL (base with path): {valid_url_base_path}")
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        # Test with path starting with /
        valid_url_abs_path = construct_and_validate_url("http://example.com/basepath", "/api/new_resource", {"id": "3"})
        print(f"Constructed valid URL (absolute path): {valid_url_abs_path}")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        invalid_url = construct_and_validate_url("http://untrusted.org", "data", {"token": "secret"})
        print(f"Constructed invalid URL: {invalid_url}") # Should not print this
    except ValueError as e:
        print(f"Error (expected for untrusted.org): {e}")

    try:
        # Test with empty path
        valid_url_empty_path = construct_and_validate_url("https://trusted.com", "", {"user": "admin"})
        print(f"Constructed valid URL (empty path): {valid_url_empty_path}")
    except ValueError as e:
        print(f"Error: {e}")
    
    try:
        # Test with empty query_params
        valid_url_empty_query = construct_and_validate_url("https://example.com", "search", {})
        print(f"Constructed valid URL (empty query): {valid_url_empty_query}")
    except ValueError as e:
        print(f"Error: {e}")
