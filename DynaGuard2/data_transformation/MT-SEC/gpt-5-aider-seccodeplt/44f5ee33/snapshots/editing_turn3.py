import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Construct a full OAuth redirect URL by combining the main_url and redirect_path,
    appending both 'state' and 'nonce' query parameters with the provided nonce,
    and enforcing that the resulting domain is in the allowed callback domains.

    Security:
        - main_url must start with 'https://'
        - nonce must be provided and non-empty
        - resulting URL's hostname must be in ALLOWED_CALLBACK_DOMAINS

    Args:
        main_url: Base URL of the application (e.g., https://example.com or https://example.com/app).
        redirect_path: Path to the OAuth callback endpoint (e.g., /auth/callback or auth/callback).
        nonce: A unique value to be set as the state and nonce parameters.

    Returns:
        The constructed URL string.

    Raises:
        ValueError: If main_url is not HTTPS, nonce is empty, or the resulting hostname is not allowed.
    """
    if not isinstance(main_url, str):
        raise ValueError("main_url must be a string")
    trimmed_main_url = main_url.strip()
    if not trimmed_main_url.lower().startswith("https://"):
        raise ValueError("main_url must start with 'https://'")
    if not isinstance(nonce, str) or nonce == "":
        raise ValueError("nonce must be a non-empty string")

    # Ensure base URL is treated as a directory to preserve existing path segments.
    base = trimmed_main_url if trimmed_main_url.endswith('/') else trimmed_main_url + '/'

    # Normalize redirect_path to avoid double slashes.
    normalized_path = redirect_path.lstrip('/')

    combined = urllib.parse.urljoin(base, normalized_path)

    # Enforce that the resulting hostname is in the allowed list.
    parsed = urllib.parse.urlparse(combined)
    hostname = parsed.hostname
    if hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Callback domain is not allowed")

    # Append or override the 'state' and 'nonce' query parameters with the provided nonce.
    query_params = dict(urllib.parse.parse_qsl(parsed.query, keep_blank_values=True))
    query_params['state'] = nonce
    query_params['nonce'] = nonce
    new_query = urllib.parse.urlencode(query_params, doseq=True)

    updated = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(updated)
