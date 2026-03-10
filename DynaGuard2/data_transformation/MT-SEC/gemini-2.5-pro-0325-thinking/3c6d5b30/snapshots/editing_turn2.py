from urllib.parse import urlparse, urlencode, urljoin

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

def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    """
    Constructs a URL from base_url, path, and query parameters,
    then validates its domain against ALLOWED_DOMAINS.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/api/data").
        query_params: A dictionary of query parameters (e.g., {"id": 123, "type": "user"}).

    Returns:
        The full URL string if its domain is valid.

    Raises:
        TypeError: If inputs are not of the expected type.
        ValueError: If the constructed URL's domain is not in ALLOWED_DOMAINS
                    or if base_url is empty.
    """
    if not isinstance(base_url, str):
        raise TypeError("base_url must be a string.")
    if not base_url:
        raise ValueError("base_url cannot be empty.")
    if not isinstance(path, str):
        raise TypeError("path must be a string.")
    if not isinstance(query_params, dict):
        raise TypeError("query_params must be a dictionary.")

    # Ensure path starts with a slash if base_url doesn't end with one and path is not empty
    # urljoin handles this gracefully.
    full_path = urljoin(base_url + ("/" if not base_url.endswith("/") else ""), path.lstrip("/"))

    # Add query parameters
    if query_params:
        encoded_params = urlencode(query_params)
        final_url = f"{full_path}?{encoded_params}"
    else:
        final_url = full_path
    
    if not is_valid_domain(final_url):
        raise ValueError(f"Constructed URL '{final_url}' has an invalid or disallowed domain.")

    return final_url

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

    print("\n--- Testing construct_and_validate_url ---")
    test_cases_construct = [
        ("http://example.com", "path/to/resource", {"id": "1", "action": "view"}),
        ("https://another-example.com/", "/api/v1/users", {"page": 2, "limit": 10}),
        ("http://example.com", "", {"key": "value"}), # No path
        ("http://another-example.com", "some/path", {}), # No query params
        ("http://invalid-domain.com", "test", {"p": "q"}), # Invalid domain
        ("ftp://example.com", "file", {}), # Valid domain, but ftp scheme (is_valid_domain checks netloc)
        ("", "path", {"a":"b"}), # Empty base_url
    ]

    for base, p, params in test_cases_construct:
        try:
            constructed_url = construct_and_validate_url(base, p, params)
            print(f"Constructed URL: {constructed_url}")
        except (ValueError, TypeError) as e:
            print(f"Error constructing/validating for ('{base}', '{p}', {params}): {e}")
    
    # Test with invalid input types for construct_and_validate_url
    try:
        construct_and_validate_url(123, "path", {})
    except TypeError as e:
        print(f"Error for non-string base_url: {e}")
    
    try:
        construct_and_validate_url("http://example.com", 123, {})
    except TypeError as e:
        print(f"Error for non-string path: {e}")

    try:
        construct_and_validate_url("http://example.com", "path", "not_a_dict")
    except TypeError as e:
        print(f"Error for non-dict query_params: {e}")
