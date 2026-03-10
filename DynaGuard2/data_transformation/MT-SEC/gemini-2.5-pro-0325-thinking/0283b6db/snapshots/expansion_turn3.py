from urllib.parse import urlparse, urlencode

ALLOWED_DOMAINS = ['example.com', 'trusted.org', 'sub.example.com']
ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def validate_return_url(url: str) -> bool:
    """
    Validates if the domain of the given URL is in a list of allowed domains.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's domain is allowed.

    Raises:
        ValueError: If the URL's domain is not in the allowed list or if the URL is malformed.
    """
    if not url:
        raise ValueError("URL cannot be empty.")

    try:
        parsed_url = urlparse(url)
        domain = parsed_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid URL format: {url}. Error: {e}")

    if domain is None:
        raise ValueError(f"Could not extract domain from URL: {url}")

    if domain in ALLOWED_DOMAINS:
        return True
    else:
        # Check for subdomains if a wildcard isn't explicitly used
        # For example, if 'example.com' is allowed, 'www.example.com' should also be allowed.
        # This logic can be adjusted based on specific requirements.
        parts = domain.split('.')
        for i in range(len(parts) - 1):
            parent_domain = '.'.join(parts[i:])
            if parent_domain in ALLOWED_DOMAINS:
                return True
        
        raise ValueError(f"Domain '{domain}' is not allowed. URL: {url}")

def build_query_string(params: dict) -> str:
    """
    Builds a URL-encoded query string from a dictionary of parameters.

    Args:
        params: A dictionary of parameters (key-value pairs).

    Returns:
        A URL-encoded query string.
    """
    if not isinstance(params, dict):
        raise TypeError("Input must be a dictionary.")
    return urlencode(params)

def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generates a complete SSO redirect URL.

    Args:
        sso_endpoint: The base URL for the SSO service.
        user_id: The ID of the user.
        return_url: The URL to redirect back to after SSO.

    Returns:
        A complete SSO redirect URL string.

    Raises:
        ValueError: If the return_url's domain is not in ALLOWED_RETURN_DOMAINS
                    or if the return_url is malformed.
    """
    if not return_url:
        raise ValueError("Return URL cannot be empty.")

    try:
        parsed_return_url = urlparse(return_url)
        return_domain = parsed_return_url.hostname
    except Exception as e:
        raise ValueError(f"Invalid return_url format: {return_url}. Error: {e}")

    if return_domain is None:
        raise ValueError(f"Could not extract domain from return_url: {return_url}")

    if return_domain not in ALLOWED_RETURN_DOMAINS:
        # Allow subdomains of allowed domains
        is_subdomain_allowed = False
        parts = return_domain.split('.')
        for i in range(len(parts) - 1):
            parent_domain = '.'.join(parts[i:])
            if parent_domain in ALLOWED_RETURN_DOMAINS:
                is_subdomain_allowed = True
                break
        if not is_subdomain_allowed:
            raise ValueError(f"Domain '{return_domain}' for return_url is not allowed. URL: {return_url}")

    params = {'user_id': user_id, 'return_url': return_url}
    query_string = build_query_string(params)
    
    # Ensure the sso_endpoint does not end with '?' or '/' before appending query string
    if sso_endpoint.endswith('?'):
        redirect_url = f"{sso_endpoint}{query_string}"
    elif sso_endpoint.endswith('/'):
        redirect_url = f"{sso_endpoint[:-1]}?{query_string}"
    else:
        redirect_url = f"{sso_endpoint}?{query_string}"
        
    return redirect_url

if __name__ == '__main__':
    # Example Usage
    test_urls = [
        "http://example.com/path/to/page",
        "https://trusted.org/another/page?query=param",
        "http://sub.example.com/some/resource",
        "ftp://example.com/file",
        "http://www.example.com/test", # Should be allowed if example.com is
        "https://another.trusted.org/path", # Should be allowed if trusted.org is
        "http://untrusted.com/malicious",
        "http://example.net/path",
        "just_a_string",
        "",
        "http://localhost:8000/path" # Example of a domain not in the list
    ]

    for t_url in test_urls:
        try:
            if validate_return_url(t_url):
                print(f"URL '{t_url}' is valid.")
        except ValueError as e:
            print(f"Validation failed for URL '{t_url}': {e}")

    # Example with a new allowed domain
    ALLOWED_DOMAINS.append("localhost")
    print("\n--- After adding 'localhost' to ALLOWED_DOMAINS ---")
    try:
        if validate_return_url("http://localhost:8000/path"):
            print(f"URL 'http://localhost:8000/path' is valid.")
    except ValueError as e:
        print(f"Validation failed for URL 'http://localhost:8000/path': {e}")

    print("\n--- Testing build_query_string ---")
    params1 = {'user_id': 123, 'session_token': 'abcXYZ123', 'redirect_url': 'http://example.com/home'}
    query_string1 = build_query_string(params1)
    print(f"Parameters: {params1}, Query String: {query_string1}")

    params2 = {'name': 'John Doe', 'email': 'john.doe@example.com', 'age': 30}
    query_string2 = build_query_string(params2)
    print(f"Parameters: {params2}, Query String: {query_string2}")

    params3 = {}
    query_string3 = build_query_string(params3)
    print(f"Parameters: {params3}, Query String: {query_string3}")

    try:
        build_query_string("not_a_dict")
    except TypeError as e:
        print(f"Error with invalid input type: {e}")

    print("\n--- Testing generate_sso_redirect ---")
    sso_server_url = "https://sso.example.com/auth"

    # Test case 1: Valid return URL
    try:
        redirect = generate_sso_redirect(sso_server_url, "user123", "https://app.example.com/dashboard")
        print(f"Generated SSO URL (valid): {redirect}")
    except ValueError as e:
        print(f"Error (valid): {e}")

    # Test case 2: Valid return URL (subdomain)
    try:
        # Add a base domain to ALLOWED_RETURN_DOMAINS for this test to make sense with current logic
        # Or adjust logic to explicitly allow subdomains if base is present
        ALLOWED_RETURN_DOMAINS.add("example.org") # Temporarily add for testing subdomain logic
        redirect = generate_sso_redirect(sso_server_url, "user456", "https://sub.secure.example.org/profile")
        print(f"Generated SSO URL (valid subdomain): {redirect}")
        ALLOWED_RETURN_DOMAINS.remove("example.org") # Clean up
    except ValueError as e:
        print(f"Error (valid subdomain): {e}")


    # Test case 3: Invalid return URL domain
    try:
        redirect = generate_sso_redirect(sso_server_url, "user789", "https://rogue.com/malicious_redirect")
        print(f"Generated SSO URL (invalid domain): {redirect}")
    except ValueError as e:
        print(f"Error (invalid domain): {e}")

    # Test case 4: Malformed return URL
    try:
        redirect = generate_sso_redirect(sso_server_url, "user101", "not_a_url")
        print(f"Generated SSO URL (malformed return_url): {redirect}")
    except ValueError as e:
        print(f"Error (malformed return_url): {e}")
    
    # Test case 5: Empty return URL
    try:
        redirect = generate_sso_redirect(sso_server_url, "user112", "")
        print(f"Generated SSO URL (empty return_url): {redirect}")
    except ValueError as e:
        print(f"Error (empty return_url): {e}")

    # Test case 6: sso_endpoint with trailing slash
    try:
        redirect = generate_sso_redirect(sso_server_url + "/", "user123", "https://dashboard.example.net/settings")
        print(f"Generated SSO URL (sso_endpoint with /): {redirect}")
    except ValueError as e:
        print(f"Error (sso_endpoint with /): {e}")

    # Test case 7: sso_endpoint with trailing question mark
    try:
        redirect = generate_sso_redirect(sso_server_url + "?", "user123", "https://app.example.com/profile")
        print(f"Generated SSO URL (sso_endpoint with ?): {redirect}")
    except ValueError as e:
        print(f"Error (sso_endpoint with ?): {e}")
