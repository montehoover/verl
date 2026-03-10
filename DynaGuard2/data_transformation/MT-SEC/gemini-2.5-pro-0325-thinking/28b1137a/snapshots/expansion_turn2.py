from urllib.parse import urlparse, urlencode

APPROVED_DOMAINS = [
    "example.com",
    "example.org",
    "example.net",
]

def validate_domain(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a list of approved domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the domain is approved.

    Raises:
        ValueError: If the domain is not approved or the URL is malformed.
    """
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid URL: {url}. Error: {e}")

    if domain is None:
        raise ValueError(f"Could not extract domain from URL: {url}")

    if domain in APPROVED_DOMAINS:
        return True
    else:
        raise ValueError(f"Domain not approved: {domain}")


def create_query_string(params: dict) -> str:
    """
    Creates a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters.

    Returns:
        A URL-encoded query string.
    """
    return urlencode(params)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path",
        "https://www.example.org/another?query=param",
        "ftp://example.net",
        "http://sub.example.com/path", # This will be approved if "example.com" is approved (subdomain)
                                      # Current logic checks exact domain match.
                                      # If subdomains should be allowed, the logic needs adjustment.
                                      # For now, this will fail as "sub.example.com" is not in APPROVED_DOMAINS.
        "http://google.com",
        "invalid-url",
    ]

    for test_url in test_urls:
        try:
            if validate_domain(test_url):
                print(f"URL '{test_url}' has an approved domain.")
        except ValueError as e:
            print(f"Validation failed for URL '{test_url}': {e}")

    print("\nTesting with a subdomain that should be approved if base domain is approved (current logic will fail):")
    # To make subdomains pass if the parent domain is in APPROVED_DOMAINS,
    # the check `if domain in APPROVED_DOMAINS:` would need to be more sophisticated, e.g.:
    # `any(domain.endswith(approved_domain) for approved_domain in APPROVED_DOMAINS)`
    # For now, "sub.example.com" will be treated as a distinct domain.
    # Let's add "sub.example.com" to APPROVED_DOMAINS for a specific test case if needed,
    # or adjust the logic as commented above for broader subdomain approval.

    # Example of a URL that should pass if "sub.example.com" is explicitly added or subdomain logic is implemented
    # For now, let's test with the current exact match logic.
    # If you want to test subdomains, you can modify APPROVED_DOMAINS or the function.

    # Example with a domain that is not in the list
    try:
        validate_domain("http://unknown.com/path")
    except ValueError as e:
        print(f"Validation failed for URL 'http://unknown.com/path': {e}")

    # Example with an invalid URL format
    try:
        validate_domain("htp:/invalid.url")
    except ValueError as e:
        print(f"Validation failed for URL 'htp:/invalid.url': {e}")

    print("\nTesting create_query_string function:")
    params1 = {"name": "John Doe", "age": "30", "city": "New York"}
    query_string1 = create_query_string(params1)
    print(f"Parameters: {params1}, Query String: '{query_string1}'")

    params2 = {"product_id": "12345", "quantity": "2", "options": ["red", "large"]}
    query_string2 = create_query_string(params2)
    print(f"Parameters: {params2}, Query String: '{query_string2}'")

    params3 = {}
    query_string3 = create_query_string(params3)
    print(f"Parameters: {params3}, Query String: '{query_string3}'")
