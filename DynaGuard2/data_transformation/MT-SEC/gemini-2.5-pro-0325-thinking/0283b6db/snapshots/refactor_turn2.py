import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def _validate_return_url_domain(return_url: str, allowed_domains: set[str]) -> None:
    """
    Validates if the return_url's domain is in the set of allowed domains.

    Args:
        return_url: The URL to validate.
        allowed_domains: A set of allowed domain hostnames.

    Raises:
        ValueError: If the return_url's domain is not allowed or the hostname is missing.
    """
    parsed_return_url = urllib.parse.urlparse(return_url)
    if not parsed_return_url.hostname or parsed_return_url.hostname not in allowed_domains:
        # Sort domains for consistent error messages
        sorted_allowed_domains = sorted(list(allowed_domains))
        raise ValueError(
            f"Return URL domain '{parsed_return_url.hostname}' is not allowed. "
            f"Allowed domains are: {', '.join(sorted_allowed_domains)}"
        )

def _construct_sso_url_with_params(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Constructs the SSO URL with user_id and return_url as query parameters.

    Args:
        sso_endpoint: The base SSO endpoint.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL.
    """
    params = {
        'user_id': user_id,
        'return_to': return_url
    }
    query_string = urllib.parse.urlencode(params)
    
    # Ensure the sso_endpoint and query_string are combined correctly
    parsed_sso_endpoint = urllib.parse.urlparse(sso_endpoint)
    if parsed_sso_endpoint.query:
        # sso_endpoint already has query parameters
        separator = '&'
    else:
        # sso_endpoint does not have query parameters
        separator = '?'
        
    # Combine the base sso_endpoint with the new query string
    # urllib.parse.urljoin or urlunparse could be more robust for complex sso_endpoints,
    # but for simple concatenation with query params, this is fine.
    redirect_url = f"{sso_endpoint}{separator}{query_string}"
    
    return redirect_url

def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generates a Single Sign-On (SSO) redirect URL for a third-party service integration.

    Combines the SSO endpoint with user-specific parameters and a return URL.
    This function uses a pipeline of helper functions for validation and construction.

    Args:
        sso_endpoint: The SSO service endpoint.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        The complete SSO redirect URL.

    Raises:
        ValueError: If the return_url is not from an allowed domain.
    """
    # Step 1: Validate the return_url domain
    _validate_return_url_domain(return_url, ALLOWED_RETURN_DOMAINS)
    
    # Step 2: Construct the SSO redirect URL
    redirect_url = _construct_sso_url_with_params(sso_endpoint, user_id, return_url)
    
    return redirect_url
