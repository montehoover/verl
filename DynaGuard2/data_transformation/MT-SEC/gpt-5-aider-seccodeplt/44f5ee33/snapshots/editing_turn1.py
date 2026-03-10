from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse

def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Construct a full OAuth redirect URL by combining the main_url and redirect_path,
    and appending the nonce as the 'state' query parameter.

    Args:
        main_url: Base URL of the application (e.g., https://example.com or https://example.com/app).
        redirect_path: Path to the OAuth callback endpoint (e.g., /auth/callback or auth/callback).
        nonce: A unique value to be set as the state parameter.

    Returns:
        The constructed URL string.
    """
    # Ensure base URL is treated as a directory to preserve existing path segments.
    base = main_url if main_url.endswith('/') else main_url + '/'

    # Normalize redirect_path to avoid double slashes.
    normalized_path = redirect_path.lstrip('/')

    combined = urljoin(base, normalized_path)

    # Append or override the 'state' query parameter with the provided nonce.
    parsed = urlparse(combined)
    query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_params['state'] = nonce
    new_query = urlencode(query_params, doseq=True)

    updated = parsed._replace(query=new_query)
    return urlunparse(updated)
