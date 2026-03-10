from urllib.parse import urlparse, urlencode, urljoin

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

def build_url_with_params(base_url: str, path: str, params: dict) -> str:
    """
    Constructs a URL with a given base URL, path, and parameters.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/api/data").
        params: A dictionary of query parameters (e.g., {"id": 123, "type": "user"}).

    Returns:
        A string representing the complete URL with encoded parameters.
    """
    if not isinstance(base_url, str):
        raise ValueError("Base URL must be a string.")
    if not isinstance(path, str):
        raise ValueError("Path must be a string.")
    if not isinstance(params, dict):
        raise ValueError("Parameters must be a dictionary.")

    # Ensure the base_url ends with a slash and path doesn't start with one for urljoin
    if not base_url.endswith('/'):
        base_url += '/'
    
    # urljoin handles path starting with '/' correctly, but for consistency we can strip it
    path = path.lstrip('/')

    # Construct the full path URL
    full_path_url = urljoin(base_url, path)

    # Encode parameters
    if params:
        query_string = urlencode(params)
        return f"{full_path_url}?{query_string}"
    else:
        return full_path_url

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

    print("\nTesting build_url_with_params:")
    base = "http://example.com/api"
    path1 = "users"
    params1 = {"id": 1, "active": "true"}
    print(f"Built URL 1: {build_url_with_params(base, path1, params1)}")

    base2 = "https://trusted.org"
    path2 = "/data/items/" # Path with trailing slash
    params2 = {"category": "books", "sort": "asc"}
    print(f"Built URL 2: {build_url_with_params(base2, path2, params2)}")

    base3 = "http://api.service.net/" # Base with trailing slash
    path3 = "search"
    params3 = {} # No parameters
    print(f"Built URL 3: {build_url_with_params(base3, path3, params3)}")

    path4 = "/v2/items" # Path with leading slash
    params4 = {"filter": "new"}
    print(f"Built URL 4: {build_url_with_params(base, path4, params4)}")

    try:
        build_url_with_params(123, "path", {})
    except ValueError as e:
        print(f"Error building URL (invalid base_url type): {e}")

    try:
        build_url_with_params("http://example.com", 123, {})
    except ValueError as e:
        print(f"Error building URL (invalid path type): {e}")

    try:
        build_url_with_params("http://example.com", "path", "not-a-dict")
    except ValueError as e:
        print(f"Error building URL (invalid params type): {e}")
