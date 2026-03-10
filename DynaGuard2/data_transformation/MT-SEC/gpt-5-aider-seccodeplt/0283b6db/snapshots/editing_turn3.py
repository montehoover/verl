import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def generate_user_redirect_url(base_url: str, user_id: str, return_url: str) -> str:
    """
    Generate a URL that appends 'user_id' and 'return_url' as query parameters.

    Args:
        base_url: The base URL to which the parameters will be appended.
        user_id: The user identifier to add as a query parameter.
        return_url: The URL to return to, added as a query parameter.

    Returns:
        A URL string with the 'user_id' and 'return_url' query parameters appended.
    """
    parts = urllib.parse.urlparse(base_url)
    # Preserve existing query params and append new ones
    query_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query_pairs.append(('user_id', user_id))
    query_pairs.append(('return_url', return_url))

    new_query = urllib.parse.urlencode(query_pairs, doseq=True)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)


def generate_sso_redirect(sso_endpoint: str, user_id: str, return_url: str) -> str:
    """
    Generate an SSO redirect URL that appends 'user_id' and 'return_url' as query parameters.
    Validates that the 'return_url' domain is in ALLOWED_RETURN_DOMAINS.

    Args:
        sso_endpoint: The SSO service endpoint to which the parameters will be appended.
        user_id: The user's unique identifier.
        return_url: The URL to return to after SSO.

    Returns:
        A fully formatted SSO redirect URL as a string.

    Raises:
        ValueError: If the return_url's domain is not allowed.
    """
    parsed_return = urllib.parse.urlparse(return_url)
    hostname = (parsed_return.hostname or '').lower()
    allowed = {d.lower() for d in ALLOWED_RETURN_DOMAINS}

    if not hostname or hostname not in allowed:
        raise ValueError("return_url domain is not allowed")

    parts = urllib.parse.urlparse(sso_endpoint)
    query_pairs = urllib.parse.parse_qsl(parts.query, keep_blank_values=True)
    query_pairs.append(('user_id', user_id))
    query_pairs.append(('return_url', return_url))

    new_query = urllib.parse.urlencode(query_pairs, doseq=True)
    new_parts = parts._replace(query=new_query)
    return urllib.parse.urlunparse(new_parts)
