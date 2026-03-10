from urllib.parse import urlencode

def create_sso_url(auth_endpoint: str, uid: str) -> str:
    """
    Generates an SSO URL with the user ID as a query parameter.

    Args:
        auth_endpoint: The base URL for the SSO endpoint.
        uid: The unique user identifier.

    Returns:
        The SSO URL with the uid appended as a query parameter.
    """
    query_params = {'uid': uid}
    # Ensure there's a '?' before query parameters if not already present
    # and handle if auth_endpoint already has query parameters.
    separator = '&' if '?' in auth_endpoint else '?'
    
    return f"{auth_endpoint}{separator}{urlencode(query_params)}"

if __name__ == '__main__':
    # Example usage:
    endpoint = "https://sso.example.com/auth"
    user_id = "user123"
    sso_url = create_sso_url(endpoint, user_id)
    print(f"Generated SSO URL: {sso_url}")

    endpoint_with_params = "https://sso.example.com/auth?existing_param=true"
    sso_url_with_existing_params = create_sso_url(endpoint_with_params, user_id)
    print(f"Generated SSO URL (with existing params): {sso_url_with_existing_params}")
