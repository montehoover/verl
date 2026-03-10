from urllib.parse import urlparse

# Define a list of trusted domains
TRUSTED_DOMAINS = [
    "example.com",
    "trusted.org",
    "another.trusted.net",
]

def validate_url(url: str) -> bool:
    """
    Validates a URL and checks if it belongs to a trusted domain.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL is valid and belongs to a trusted domain, False otherwise.

    Raises:
        ValueError: If the URL is malformed or has an invalid scheme.
    """
    if not isinstance(url, str):
        raise TypeError("URL must be a string.")

    try:
        parsed_url = urlparse(url)
    except Exception as e: # Catch any parsing errors, though urlparse is quite robust
        raise ValueError(f"URL parsing failed: {e}")

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError("URL is malformed. It must include a scheme (e.g., http, https) and a domain.")

    if parsed_url.scheme not in ["http", "https"]:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only 'http' and 'https' are allowed.")

    # Extract the domain (netloc)
    domain = parsed_url.netloc
    # Remove port if present (e.g., example.com:8080 -> example.com)
    if ":" in domain:
        domain = domain.split(":")[0]

    if domain in TRUSTED_DOMAINS:
        return True
    
    # Check for subdomains of trusted domains if needed, e.g. api.example.com
    # For simplicity, this example only checks exact domain matches.
    # If subdomains are allowed, the logic would be:
    # for trusted_domain in TRUSTED_DOMAINS:
    #     if domain == trusted_domain or domain.endswith("." + trusted_domain):
    #         return True
            
    return False

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "https://example.com/path/to/resource",
        "http://trusted.org",
        "https://sub.example.com/another?query=param", # This will be false unless subdomains are handled
        "ftp://untrusted.com",
        "https://another.trusted.net/secure",
        "https://malicious-site.com",
        "http://example.com:8080/path",
        "justadomain.com/path", # Invalid, no scheme
        "http:///path-only", # Invalid, no domain
        "https://", # Invalid
        "", # Invalid
        None, # Will raise TypeError
        123, # Will raise TypeError
    ]

    for test_url in test_urls:
        try:
            is_valid = validate_url(test_url)
            print(f"URL: '{test_url}', Valid and Trusted: {is_valid}")
        except (ValueError, TypeError) as e:
            print(f"URL: '{test_url}', Error: {e}")

    print("\nTesting with a trusted subdomain (requires subdomain logic in validate_url to pass):")
    # To make this pass, you would need to adjust TRUSTED_DOMAINS or the logic in validate_url
    # For example, add "sub.trusted.org" to TRUSTED_DOMAINS or implement subdomain checking.
    try:
        is_valid = validate_url("https://sub.trusted.org/api")
        print(f"URL: 'https://sub.trusted.org/api', Valid and Trusted: {is_valid}")
    except (ValueError, TypeError) as e:
        print(f"URL: 'https://sub.trusted.org/api', Error: {e}")
