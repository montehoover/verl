import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _validate_absolute_http_url(url: str, subject: str) -> urllib.parse.ParseResult:
    """
    Validate that the given URL is an absolute HTTP(S) URL with a host.

    Args:
        url: The URL string to validate.
        subject: A label used in error messages to indicate what is being validated
                 (e.g., "main_url" or "callback URL").

    Returns:
        The parsed URL as a urllib.parse.ParseResult.

    Raises:
        ValueError: If the URL is not a valid absolute HTTP(S) URL.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        if subject == "main_url":
            raise ValueError("Invalid main_url: must include a valid http(s) scheme and host")
        else:
            raise ValueError("Invalid callback URL: must include a valid http(s) scheme and host")
    return parsed


def _construct_callback_url(main_url: str, redirect_path: str) -> str:
    """
    Construct the callback URL from a main URL and a redirect path or absolute URL.

    If redirect_path is absolute, it is returned as-is. Otherwise, it is joined
    with the origin (scheme + host) of main_url.

    Args:
        main_url: The main URL of the application.
        redirect_path: The path or absolute URL for the OAuth callback.

    Returns:
        The constructed callback URL as a string.
    """
    redirect_parsed = urllib.parse.urlparse(redirect_path)
    if redirect_parsed.scheme and redirect_parsed.netloc:
        return redirect_path

    base_parsed = urllib.parse.urlparse(main_url)
    base_origin = f"{base_parsed.scheme}://{base_parsed.netloc}/"
    return urllib.parse.urljoin(base_origin, redirect_path.lstrip("/"))


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
    # Validate the main URL first.
    _validate_absolute_http_url(main_url, "main_url")

    # Construct the callback URL from inputs.
    callback_url = _construct_callback_url(main_url, redirect_path)

    # Validate the constructed callback URL.
    cb_parsed = _validate_absolute_http_url(callback_url, "callback URL")

    # Add or replace the 'state' query parameter with the provided nonce.
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
