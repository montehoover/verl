import urllib.parse # Keep this, as urlparse is part of it.

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Generates an SSO URL with the user ID and redirect URL as query parameters,
    validating the redirect URL protocol and domain.

    Args:
        auth_endpoint: The base URL for the SSO endpoint.
        uid: The unique user identifier.
        redirect_url: The URL to redirect to after successful authentication.
                      Must start with "https://" and belong to an allowed domain.

    Returns:
        The SSO URL with uid and redirect_url appended as query parameters.
    
    Raises:
        ValueError: If redirect_url does not start with "https://",
                    or if its domain is not in ALLOWED_RETURN_DOMAINS.
    """
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure redirect_url provided. URL must start with 'https://'.")

    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    redirect_domain = parsed_redirect_url.hostname
    if redirect_domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(f"Redirect URL domain '{redirect_domain}' is not allowed.")
    
    query_params = {'uid': uid, 'return_url': redirect_url}
    # Ensure there's a '?' before query parameters if not already present
    # and handle if auth_endpoint already has query parameters.
    separator = '&' if '?' in auth_endpoint else '?'
    
    return f"{auth_endpoint}{separator}{urllib.parse.urlencode(query_params)}"

if __name__ == '__main__':
    # Example usage:
    endpoint = "https://sso.example.com/auth"
    user_id = "user123"
    
    # Valid redirect URL (protocol and domain)
    valid_redirect_allowed_domain = "https://app.example.com/dashboard"
    sso_url = create_sso_url(endpoint, user_id, valid_redirect_allowed_domain)
    print(f"Generated SSO URL (valid): {sso_url}")

    endpoint_with_params = "https://sso.example.com/auth?existing_param=true"
    sso_url_with_existing_params = create_sso_url(endpoint_with_params, user_id, valid_redirect_allowed_domain)
    print(f"Generated SSO URL (with existing params, valid): {sso_url_with_existing_params}")

    # Example of invalid redirect_url (insecure protocol)
    invalid_protocol_redirect = "http://app.example.com/dashboard"
    try:
        create_sso_url(endpoint, user_id, invalid_protocol_redirect)
    except ValueError as e:
        print(f"Error (insecure protocol): {e}")

    # Example of invalid redirect_url (disallowed domain)
    disallowed_domain_redirect = "https://untrusted.com/path"
    try:
        create_sso_url(endpoint, user_id, disallowed_domain_redirect)
    except ValueError as e:
        print(f"Error (disallowed domain): {e}")
    
    # Example of valid redirect URL (another allowed domain)
    valid_redirect_another_allowed_domain = "https://dashboard.example.net/profile"
    sso_url_another = create_sso_url(endpoint, user_id, valid_redirect_another_allowed_domain)
    print(f"Generated SSO URL (another valid domain): {sso_url_another}")
