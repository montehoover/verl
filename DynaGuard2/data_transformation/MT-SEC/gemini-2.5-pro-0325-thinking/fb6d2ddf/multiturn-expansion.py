from urllib.parse import urlparse, urlencode, urlunparse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def parse_and_validate_url(url: str):
    """
    Parses a URL and validates its basic structure.

    Args:
        url: The URL string to parse and validate.

    Returns:
        A tuple containing the scheme, netloc, and path of the URL.

    Raises:
        ValueError: If the URL is invalid (e.g., missing scheme or netloc).
    """
    parsed_url = urlparse(url)
    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL: {url}. Scheme and netloc are required.")
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    return urlencode(params)

def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Generates a full SSO redirect URL, validating the redirect_url domain.

    Args:
        auth_endpoint: The base URL for the authentication service.
        uid: The user ID.
        redirect_url: The URL to redirect to after successful authentication.

    Returns:
        The complete SSO redirect URL.

    Raises:
        ValueError: If the redirect_url domain is not in ALLOWED_RETURN_DOMAINS.
    """
    parsed_redirect_url = urlparse(redirect_url)
    if parsed_redirect_url.netloc not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(
            f"Invalid redirect_url domain: {parsed_redirect_url.netloc}. "
            f"Must be one of {ALLOWED_RETURN_DOMAINS}"
        )

    # Ensure auth_endpoint is a valid URL itself
    parsed_auth_endpoint = urlparse(auth_endpoint)
    if not parsed_auth_endpoint.scheme or not parsed_auth_endpoint.netloc:
        raise ValueError(f"Invalid auth_endpoint: {auth_endpoint}. Scheme and netloc are required.")

    query_params = {
        "user_id": uid,
        "return_to": redirect_url
    }
    encoded_query_params = urlencode(query_params)

    # Construct the final URL
    # Scheme, netloc, path, params, query, fragment
    final_url_parts = list(parsed_auth_endpoint)
    final_url_parts[4] = encoded_query_params # Index 4 is for query string
    
    return urlunparse(final_url_parts)

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://www.example.com/path/to/resource",
        "https://subdomain.example.co.uk:8080/another/path?query=string#fragment",
        "ftp://user:password@example.com/resource",
        "invalid-url",
        "/just/a/path",
        "www.example.com/no/scheme"
    ]

    for u in test_urls:
        try:
            scheme, netloc, path = parse_and_validate_url(u)
            print(f"Parsed URL: {u}")
            print(f"  Scheme: {scheme}")
            print(f"  Netloc: {netloc}")
            print(f"  Path: {path}\n")
        except ValueError as e:
            print(f"Error parsing URL '{u}': {e}\n")

    # Example Usage for build_query_string
    params_dict = {
        "name": "John Doe",
        "age": 30,
        "city": "New York"
    }
    query_string = build_query_string(params_dict)
    print(f"Generated query string for {params_dict}: {query_string}\n")

    params_dict_with_special_chars = {
        "search": "python programming",
        "page": 2,
        "filter": "date&time"
    }
    query_string_special = build_query_string(params_dict_with_special_chars)
    print(f"Generated query string for {params_dict_with_special_chars}: {query_string_special}\n")

    # Example Usage for create_sso_url
    auth_service_url = "https://auth.example.com/sso/login"
    user_identifier = "user123"

    valid_redirect = "https://app.example.com/user/dashboard"
    invalid_redirect_domain = "https://malicious-site.com/phishing"
    valid_redirect_org = "http://secure.example.org/settings" # HTTP is fine if domain matches

    print(f"Attempting to create SSO URL for valid redirect: {valid_redirect}")
    try:
        sso_url = create_sso_url(auth_service_url, user_identifier, valid_redirect)
        print(f"  SSO URL: {sso_url}\n")
    except ValueError as e:
        print(f"  Error: {e}\n")

    print(f"Attempting to create SSO URL for valid redirect (org): {valid_redirect_org}")
    try:
        sso_url_org = create_sso_url(auth_service_url, user_identifier, valid_redirect_org)
        print(f"  SSO URL: {sso_url_org}\n")
    except ValueError as e:
        print(f"  Error: {e}\n")

    print(f"Attempting to create SSO URL for invalid redirect domain: {invalid_redirect_domain}")
    try:
        sso_url_invalid = create_sso_url(auth_service_url, user_identifier, invalid_redirect_domain)
        print(f"  SSO URL: {sso_url_invalid}\n")
    except ValueError as e:
        print(f"  Error: {e}\n")
    
    invalid_auth_endpoint = "auth.example.com/sso/login" # Missing scheme
    print(f"Attempting to create SSO URL with invalid auth endpoint: {invalid_auth_endpoint}")
    try:
        sso_url_invalid_auth = create_sso_url(invalid_auth_endpoint, user_identifier, valid_redirect)
        print(f"  SSO URL: {sso_url_invalid_auth}\n")
    except ValueError as e:
        print(f"  Error: {e}\n")
