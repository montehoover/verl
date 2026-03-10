def build_secure_url(base_url: str, path: str, security_token: str) -> str:
    """
    Constructs a URL from a base URL, a path, and a security token.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/users").
        security_token: The security token to append as a query parameter.

    Returns:
        The full URL with the security token as a query parameter.
    """
    return f"{base_url}{path}?token={security_token}"
