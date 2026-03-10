import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generates a Single Sign-On (SSO) redirect URL for a third-party service integration.

    Combines the SSO endpoint with user-specific parameters and a return URL.

    Args:
        sso_endpoint: The SSO service endpoint.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL.

    Raises:
        ValueError: If the return_url is not from an allowed domain.
    """
    parsed_return_url = urllib.parse.urlparse(return_url)
    if parsed_return_url.hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError(
            f"Return URL domain '{parsed_return_url.hostname}' is not allowed. "
            f"Allowed domains are: {', '.join(ALLOWED_RETURN_DOMAINS)}"
        )

    params = {
        'user': user_id,
        'return': return_url
    }
    query_string = urllib.parse.urlencode(params)
    
    # Ensure there's a '?' before query parameters if sso_endpoint doesn't have one
    # and doesn't end with '/' (in which case urljoin would handle it better,
    # but for simple query appending, this is fine).
    # However, urllib.parse.urlunparse or url_join might be more robust for complex sso_endpoints.
    # For this specific case, simple concatenation with '?' or '&' based on existing '?' is common.
    
    # A more robust way to add query parameters:
    parsed_sso_endpoint = urllib.parse.urlparse(sso_endpoint)
    # If sso_endpoint already has query params, new params will be added correctly by urlencode
    # if we parse existing query, update dict, then re-encode.
    # For simplicity here, we assume sso_endpoint is a base URL without query params or we just append.
    
    # If sso_endpoint might have its own query parameters, a more robust approach would be:
    # sso_parts = list(urllib.parse.urlparse(sso_endpoint))
    # existing_query_params = urllib.parse.parse_qs(sso_parts[4])
    # all_params = {**existing_query_params, **params} # Python 3.5+ for dict unpacking
    # sso_parts[4] = urllib.parse.urlencode(all_params, doseq=True)
    # final_url = urllib.parse.urlunparse(sso_parts)
    # return final_url
    
    # Given the example output, it seems like a simple append is expected.
    # The example output "https://sso.example.com/auth?user=user123&return=https%3A%2F%2Fapp.example.com%2Fdashboard"
    # implies sso_endpoint is "https://sso.example.com/auth"
    
    if '?' in sso_endpoint:
        separator = '&'
    else:
        separator = '?'
        
    return f"{sso_endpoint}{separator}{query_string}"
