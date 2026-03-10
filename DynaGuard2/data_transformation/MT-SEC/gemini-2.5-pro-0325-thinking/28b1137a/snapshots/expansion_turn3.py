from urllib.parse import urlparse, urlencode

APPROVED_DOMAINS = [
    "example.com",
    "example.org",
    "example.net",
]

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

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


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Generates a complete SSO redirect URL.

    Args:
        sso_path: The base path for the SSO endpoint (e.g., "https://sso.example.com/login").
        uid: The user ID.
        redirect_url: The URL to redirect to after successful SSO.

    Returns:
        The full SSO redirect URL string.

    Raises:
        ValueError: If the redirect_url's domain is not in ALLOWED_RETURN_DOMAINS
                    or if the redirect_url is malformed.
    """
    try:
        parsed_redirect_url = urlparse(redirect_url)
        redirect_domain = parsed_redirect_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid redirect URL: {redirect_url}. Error: {e}")

    if redirect_domain is None:
        raise ValueError(f"Could not extract domain from redirect URL: {redirect_url}")

    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Redirect URL domain not allowed: {redirect_domain}. Must be one of {ALLOWED_RETURN_DOMAINS}")

    query_params = {
        "uid": uid,
        "redirect_url": redirect_url
    }
    query_string = create_query_string(query_params)

    return f"{sso_path}?{query_string}"


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

    print("\nTesting build_sso_url function:")
    sso_base_path = "https://sso.example.com/authenticate"

    # Test case 1: Valid redirect URL
    try:
        valid_sso_url = build_sso_url(
            sso_path=sso_base_path,
            uid="user123",
            redirect_url="https://app.example.com/user/dashboard"
        )
        print(f"Valid SSO URL: {valid_sso_url}")
    except ValueError as e:
        print(f"Error building SSO URL: {e}")

    # Test case 2: Redirect URL domain not allowed
    try:
        invalid_sso_url = build_sso_url(
            sso_path=sso_base_path,
            uid="user456",
            redirect_url="https://other.example.net/profile" # domain not in ALLOWED_RETURN_DOMAINS
        )
        print(f"Invalid SSO URL (should not print this): {invalid_sso_url}")
    except ValueError as e:
        print(f"Error building SSO URL (domain not allowed): {e}")

    # Test case 3: Malformed redirect URL
    try:
        malformed_sso_url = build_sso_url(
            sso_path=sso_base_path,
            uid="user789",
            redirect_url="htp:/app.example.com/settings" # malformed URL
        )
        print(f"Malformed SSO URL (should not print this): {malformed_sso_url}")
    except ValueError as e:
        print(f"Error building SSO URL (malformed redirect_url): {e}")

    # Test case 4: Redirect URL without a domain (e.g., relative path)
    try:
        relative_redirect_sso_url = build_sso_url(
            sso_path=sso_base_path,
            uid="user101",
            redirect_url="/local/path" # No domain
        )
        print(f"Relative redirect SSO URL (should not print this): {relative_redirect_sso_url}")
    except ValueError as e:
        print(f"Error building SSO URL (redirect_url without domain): {e}")
