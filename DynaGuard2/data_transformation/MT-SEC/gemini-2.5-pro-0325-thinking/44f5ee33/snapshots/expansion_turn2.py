from urllib.parse import urlparse

ALLOWED_DOMAINS = ["example.com", "example.org", "test.com"]

def validate_url_domain(url: str):
    """
    Validates if the domain of the given URL is in the ALLOWED_DOMAINS list.

    Args:
        url: The URL string to validate.

    Raises:
        ValueError: If the URL's domain is not in ALLOWED_DOMAINS 
                    or if the URL is malformed.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Common case: remove www. prefix if present for comparison
        if domain.startswith("www."):
            domain = domain[4:]
        
        if not domain:
            raise ValueError(f"Could not parse domain from URL: {url}")

        if domain not in ALLOWED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not an allowed domain.")
    except Exception as e:
        # Catch potential errors from urlparse itself or re-raise our ValueErrors
        if isinstance(e, ValueError):
            raise
        raise ValueError(f"Invalid URL format: {url}") from e

def combine_base_with_path(base: str, path: str) -> str:
    """
    Combines a base URL with a path, handling leading/trailing slashes.

    Args:
        base: The base URL string.
        path: The path string to append.

    Returns:
        The full combined URL string.
    """
    if base.endswith('/'):
        base = base[:-1]
    if path.startswith('/'):
        path = path[1:]
    return f"{base}/{path}"

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://www.example.org/another?query=param",
        "ftp://test.com/resource",
        "http://sub.example.com/path", # This will fail as "sub.example.com" is not in ALLOWED_DOMAINS
        "http://invalid-domain.net/page",
        "not_a_url",
        "http://example.com", # Test without path
        "https://www.test.com", # Test with www and without path
    ]

    for url in test_urls:
        try:
            validate_url_domain(url)
            print(f"URL '{url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{url}': {e}")

    print("\nTesting with a different set of allowed domains:")
    # Example of how to change allowed domains if needed (though typically it's a global constant)
    original_allowed_domains = list(ALLOWED_DOMAINS) # make a copy
    ALLOWED_DOMAINS.clear()
    ALLOWED_DOMAINS.extend(["another.com", "specific.io"])
    print(f"New allowed domains: {ALLOWED_DOMAINS}")

    new_test_urls = [
        "http://another.com/path",
        "https://www.specific.io/resource",
        "http://example.com/page" # This should fail now
    ]
    for url in new_test_urls:
        try:
            validate_url_domain(url)
            print(f"URL '{url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{url}': {e}")
    
    # Restore original for other potential tests if this were a module
    ALLOWED_DOMAINS.clear()
    ALLOWED_DOMAINS.extend(original_allowed_domains)

    print("\nTesting combine_base_with_path function:")
    test_combinations = [
        ("http://example.com", "path/to/resource"),
        ("http://example.com/", "path/to/resource"),
        ("http://example.com", "/path/to/resource"),
        ("http://example.com/", "/path/to/resource"),
        ("http://example.com/api", "v1/users"),
        ("http://example.com/api/", "/v1/users/"), # Path's trailing slash should be preserved if intended
    ]
    for base, path_segment in test_combinations:
        combined_url = combine_base_with_path(base, path_segment)
        print(f"Base: '{base}', Path: '{path_segment}' -> Combined: '{combined_url}'")
