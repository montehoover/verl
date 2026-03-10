import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Build a secure OAuth callback URL by combining a HTTPS base URL and a callback path,
    and append the given state parameter as a query string.

    - base_url must start with "https://", otherwise a ValueError is raised.
    - base_url's hostname must be in ALLOWED_CALLBACK_DOMAINS, otherwise a ValueError is raised.
    - Ensures exactly one "/" between base_url and callback_path.
    - If callback_path is empty or None, uses base_url as-is (plus the state parameter).
    """
    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise ValueError("base_url must start with 'https://'")

    parsed_base = urllib.parse.urlsplit(base_url)
    hostname = (parsed_base.hostname or "").lower()
    if hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("base_url domain is not allowed")

    base = base_url.rstrip("/")
    if callback_path:
        path = str(callback_path).lstrip("/")
        combined = f"{base}/{path}"
    else:
        combined = base

    parsed = urllib.parse.urlsplit(combined)
    query_params = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    query_params.append(("state", str(state)))
    new_query = urllib.parse.urlencode(query_params)

    new_parsed = parsed._replace(query=new_query)
    return urllib.parse.urlunsplit(new_parsed)
