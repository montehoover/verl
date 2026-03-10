from urllib.parse import urlparse, urlunparse

APPROVED_DOMAINS = {"example.com", "google.com", "openai.com"}
APPROVED_ENDPOINTS = {"/path/page", "/search", "/maps", "/test", "/another/page", "/"} # Added "/" for root paths

def validate_url_domain(url: str, expected_endpoint: str) -> str:
    """
    Validates if the domain of the given URL is in a predefined set of approved domains
    and if its path matches the expected endpoint, which must also be an approved endpoint.

    Args:
        url: The URL string to validate.
        expected_endpoint: The expected path/endpoint of the URL.

    Returns:
        The validated and reconstructed URL string if all checks pass.

    Raises:
        ValueError: If the URL's domain or endpoint is not approved, if the URL's path
                    does not match the expected endpoint, or if the URL is malformed.
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

        actual_path = parsed_url.path
        if not actual_path: # Treat empty path in URL as "/"
            actual_path = "/"
        
        if actual_path != expected_endpoint:
            raise ValueError(f"URL path '{actual_path}' does not match expected endpoint '{expected_endpoint}'.")

        if expected_endpoint not in APPROVED_ENDPOINTS:
            raise ValueError(f"Endpoint '{expected_endpoint}' is not an approved endpoint.")
        
        # Reconstruct the URL to ensure it's well-formed and includes query parameters
        # Use original netloc from parsed_url to preserve port if it was there and valid
        # Use actual_path which is now validated against expected_endpoint
        validated_url_parts = (
            parsed_url.scheme,
            parsed_url.netloc, # Keep original netloc to preserve port
            actual_path,
            parsed_url.params, # Usually empty for http/https
            parsed_url.query,
            parsed_url.fragment
        )
        validated_url = urlunparse(validated_url_parts)
        return validated_url

    except Exception as e: # Catch potential errors from urlparse or our own logic
        if isinstance(e, ValueError): # Re-raise our specific ValueErrors
            raise
        # Wrap other exceptions (e.g. from urlparse on a very malformed URL) in a ValueError
        raise ValueError(f"Invalid URL or unable to parse: {url}. Original error: {e}")

if __name__ == '__main__':
    # Test cases
    test_scenarios = [
        ("http://example.com/path/page", "/path/page", True),
        ("https://www.google.com/search?q=python&lang=en", "/search", True),
        ("ftp://openai.com/maps", "/maps", True), # Assuming ftp is a valid scheme for urlparse
        ("http://example.com", "/", True), # Test root path
        ("http://example.com/", "/", True), # Test root path with trailing slash

        # Domain failures
        ("http://sub.example.com/another/page", "/another/page", False), # "sub.example.com" not in APPROVED_DOMAINS
        ("http://example.org/path/page", "/path/page", False), # "example.org" not in APPROVED_DOMAINS
        ("https://www.anotherdomain.net/search", "/search", False), # "anotherdomain.net" not in APPROVED_DOMAINS

        # Endpoint mismatch failures
        ("http://example.com/actual/path", "/expected/path", False), # Path mismatch

        # Endpoint not approved failures
        ("http://example.com/unapproved/endpoint", "/unapproved/endpoint", False), # Endpoint not in APPROVED_ENDPOINTS

        # Malformed URL / No domain
        ("example.com/path/page", "/path/page", False), # No scheme, urlparse might treat "example.com" as path
        ("http://localhost:8000/test", "/test", False), # "localhost" not in APPROVED_DOMAINS

        # Valid cases with ports
        ("https://google.com:443/maps?id=123", "/maps", True),
        ("http://www.example.com:8080/test", "/test", True),

        # Query parameters and fragments
        ("http://example.com/path/page?name=test#section", "/path/page", True),
        ("https://google.com/search?q=ai+tools", "/search", True),
    ]

    print(f"Approved domains: {APPROVED_DOMAINS}")
    print(f"Approved endpoints: {APPROVED_ENDPOINTS}\n")

    for url_to_test, endpoint_to_check, should_pass in test_scenarios:
        print(f"Testing URL: '{url_to_test}' with expected endpoint: '{endpoint_to_check}'")
        try:
            validated_url = validate_url_domain(url_to_test, endpoint_to_check)
            if should_pass:
                print(f"Validation successful. Returned URL: '{validated_url}'")
            else:
                print(f"ERROR: Validation passed for '{url_to_test}' but was expected to fail.")
        except ValueError as e:
            if should_pass:
                print(f"ERROR: Validation failed for '{url_to_test}': {e}. Was expected to pass.")
            else:
                print(f"Validation failed as expected: {e}")
        print("-" * 30)

    # Example of a very malformed URL (domain extraction will fail)
    malformed_url = "http:///::::"
    malformed_endpoint = "/test"
    print(f"Testing malformed URL: '{malformed_url}' with expected endpoint: '{malformed_endpoint}'")
    try:
        validate_url_domain(malformed_url, malformed_endpoint)
        print(f"ERROR: Validation passed for malformed URL '{malformed_url}' but was expected to fail.")
    except ValueError as e:
        print(f"Validation failed for malformed URL as expected: {e}")
    print("-" * 30)
