import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Construct an OAuth callback URL by combining application_url and callback_route,
    appending the 'state' query parameter, and validating the domain against ALLOWED_CALLBACK_DOMAINS.
    Raises:
        ValueError: if the hostname of the resulting URL is not in ALLOWED_CALLBACK_DOMAINS.
    """
    application_url = application_url or ""
    callback_route = callback_route or ""

    base_parsed = urllib.parse.urlparse(application_url)
    base_for_join = urllib.parse.urlunparse((
        base_parsed.scheme,
        base_parsed.netloc,
        base_parsed.path or "/",
        "",
        "",
        "",
    ))

    combined_url = urllib.parse.urljoin(base_for_join, callback_route)
    parsed = urllib.parse.urlparse(combined_url)

    hostname = parsed.hostname
    if not hostname or hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("OAuth callback domain is not allowed")

    # Merge existing query params and set/override state
    query_pairs = [
        (k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() != "state"
    ]
    query_pairs.append(("state", "" if token_state is None else str(token_state)))
    new_query = urllib.parse.urlencode(query_pairs, doseq=True)

    return urllib.parse.urlunparse(parsed._replace(query=new_query))
