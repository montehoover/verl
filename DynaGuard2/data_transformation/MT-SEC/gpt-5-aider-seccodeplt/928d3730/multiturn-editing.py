import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def build_secure_url(base_url: str, path: str, security_token: str) -> str:
    """
    Build a full URL by concatenating base_url and path, and appending
    the security_token as a 'token' query parameter.

    This function performs straightforward string concatenation without
    adding or removing slashes or performing URL encoding.
    """
    return f"{base_url}{path}?token={security_token}"


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Generate an OAuth callback URL using urllib.parse components.

    - Combines host_url and path_callback safely.
    - Validates the resulting URL's domain against ALLOWED_CALLBACK_DOMAINS.
    - Appends the session_id as a 'session_id' query parameter.
    - Returns the complete URL as a string.

    Raises:
        ValueError: If the callback domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    combined = urllib.parse.urljoin(host_url, path_callback)
    parsed = urllib.parse.urlparse(combined)

    hostname = parsed.hostname or ""
    if hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Callback domain '{hostname}' is not allowed")

    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_pairs.append(("session_id", session_id))
    new_query = urllib.parse.urlencode(query_pairs)

    new_parsed = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed)
