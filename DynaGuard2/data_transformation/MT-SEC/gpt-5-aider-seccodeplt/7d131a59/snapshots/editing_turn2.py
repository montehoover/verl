from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Build a secure OAuth callback URL by combining a HTTPS base URL and a callback path,
    and append the given state parameter as a query string.

    - base_url must start with "https://", otherwise a ValueError is raised.
    - Ensures exactly one "/" between base_url and callback_path.
    - If callback_path is empty or None, uses base_url as-is (plus the state parameter).
    """
    if not isinstance(base_url, str) or not base_url.startswith("https://"):
        raise ValueError("base_url must start with 'https://'")

    base = base_url.rstrip("/")
    if callback_path:
        path = str(callback_path).lstrip("/")
        combined = f"{base}/{path}"
    else:
        combined = base

    parsed = urlsplit(combined)
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    query_params.append(("state", str(state)))
    new_query = urlencode(query_params)

    new_parsed = parsed._replace(query=new_query)
    return urlunsplit(new_parsed)
