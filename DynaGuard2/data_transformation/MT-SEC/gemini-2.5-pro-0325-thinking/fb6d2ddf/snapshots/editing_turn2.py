from urllib.parse import urlencode

def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Generates an SSO URL with the user ID and redirect URL as query parameters.

    Args:
        auth_endpoint: The base URL for the SSO endpoint.
        uid: The unique user identifier.
        redirect_url: The URL to redirect to after successful authentication.
                      Must start with "https://".

    Returns:
        The SSO URL with uid and redirect_url appended as query parameters.
    
    Raises:
        ValueError: If redirect_url does not start with "https://".
    """
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure redirect_url provided. URL must start with 'https://'.")
    
    query_params = {'uid': uid, 'return_url': redirect_url}
    # Ensure there's a '?' before query parameters if not already present
    # and handle if auth_endpoint already has query parameters.
    separator = '&' if '?' in auth_endpoint else '?'
    
    return f"{auth_endpoint}{separator}{urlencode(query_params)}"

if __name__ == '__main__':
    # Example usage:
    endpoint = "https://sso.example.com/auth"
    user_id = "user123"
    valid_redirect = "https://example.com/dashboard"
    
    sso_url = create_sso_url(endpoint, user_id, valid_redirect)
    print(f"Generated SSO URL: {sso_url}")

    endpoint_with_params = "https://sso.example.com/auth?existing_param=true"
    sso_url_with_existing_params = create_sso_url(endpoint_with_params, user_id, valid_redirect)
    print(f"Generated SSO URL (with existing params): {sso_url_with_existing_params}")

    # Example of invalid redirect_url
    invalid_redirect = "http://example.com/danger"
    try:
        create_sso_url(endpoint, user_id, invalid_redirect)
    except ValueError as e:
        print(f"Error for invalid redirect: {e}")
