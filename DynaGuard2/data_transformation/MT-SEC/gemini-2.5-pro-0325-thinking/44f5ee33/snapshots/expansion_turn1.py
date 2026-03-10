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
