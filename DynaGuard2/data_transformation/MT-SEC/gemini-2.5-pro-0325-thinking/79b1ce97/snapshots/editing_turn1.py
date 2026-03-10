from urllib.parse import urlparse

APPROVED_DOMAINS = {"example.com", "google.com", "openai.com"}

def validate_url_domain(url: str):
    """
    Validates if the domain of the given URL is in a predefined set of approved domains.

    Args:
        url: The URL string to validate.

    Raises:
        ValueError: If the URL's domain is not in the approved list or if the URL is malformed.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        # Remove port if present, e.g., "example.com:8080" -> "example.com"
        if ":" in domain:
            domain = domain.split(":")[0]
            
        # Handle www. prefix, e.g. "www.example.com" -> "example.com"
        if domain.startswith("www."):
            domain = domain[4:]

        if not domain:
            raise ValueError(f"Could not extract domain from URL: {url}")

        if domain not in APPROVED_DOMAINS:
            raise ValueError(f"Domain '{domain}' is not an approved domain.")
        
        # If we reach here, the domain is approved
        print(f"URL '{url}' with domain '{domain}' is approved.")

    except Exception as e: # Catch potential errors from urlparse or our own logic
        if isinstance(e, ValueError): # Re-raise our specific ValueErrors
            raise
        # Wrap other exceptions (e.g. from urlparse on a very malformed URL) in a ValueError
        raise ValueError(f"Invalid URL or unable to parse: {url}. Original error: {e}")

if __name__ == '__main__':
    # Test cases
    test_urls = [
        "http://example.com/path/page",
        "https://www.google.com/search?q=python",
        "ftp://openai.com",
        "http://sub.example.com/another/page", # This should fail as "sub.example.com" is not in APPROVED_DOMAINS
        "http://example.org/path", # This should fail
        "https://www.anotherdomain.net", # This should fail
        "example.com", # This should fail (not a full URL, urlparse might handle it differently)
        "http://localhost:8000", # This should fail
        "https://google.com:443/maps",
        "http://www.example.com:8080/test"
    ]

    print(f"Approved domains: {APPROVED_DOMAINS}\n")

    for url_to_test in test_urls:
        try:
            validate_url_domain(url_to_test)
        except ValueError as e:
            print(f"Validation failed for '{url_to_test}': {e}")
        print("-" * 20)

    # Example of a very malformed URL
    malformed_url = "http:///::::"
    try:
        validate_url_domain(malformed_url)
    except ValueError as e:
        print(f"Validation failed for '{malformed_url}': {e}")
    print("-" * 20)
