import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Build the full OAuth callback URL.

    Args:
        main_url: The main URL of the application.
        redirect_path: Path or URL for the OAuth callback.
        nonce: State parameter to include for verifying the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the resulting callback URL is invalid or points to an unauthorized domain.
    """
    # Parse the main URL and perform basic validation.
    base_parsed = urllib.parse.urlparse(main_url)
    if base_parsed.scheme not in {"http", "https"} or not base_parsed.netloc:
        raise ValueError("Invalid main_url: must include a valid http(s) scheme and host")

    # If redirect_path is an absolute URL, use it as-is; otherwise, join with the base (scheme + netloc).
    redirect_parsed = urllib.parse.urlparse(redirect_path)
    if redirect_parsed.scheme and redirect_parsed.netloc:
        callback_url = redirect_path
    else:
        base_origin = f"{base_parsed.scheme}://{base_parsed.netloc}/"
        callback_url = urllib.parse.urljoin(base_origin, redirect_path.lstrip("/"))

    # Add or replace the 'state' query parameter with the provided nonce.
    cb_parsed = urllib.parse.urlparse(callback_url)
    if cb_parsed.scheme not in {"http", "https"} or not cb_parsed.netloc:
        raise ValueError("Invalid callback URL: must include a valid http(s) scheme and host")

    query_items = urllib.parse.parse_qsl(cb_parsed.query, keep_blank_values=True)
    query_dict = dict(query_items)
    query_dict["state"] = nonce
    new_query = urllib.parse.urlencode(query_dict, doseq=True)
    cb_parsed = cb_parsed._replace(query=new_query)
    final_url = urllib.parse.urlunparse(cb_parsed)

    # Validate the callback's domain against the allowlist.
    hostname = cb_parsed.hostname
    if not hostname or hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Unauthorized callback domain")

    return final_url
