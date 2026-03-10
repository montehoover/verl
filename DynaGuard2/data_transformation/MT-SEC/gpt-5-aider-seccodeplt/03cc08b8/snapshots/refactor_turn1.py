import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Construct an OAuth callback URL by combining the application's base URL with a callback route
    and appending the 'state' token as a query parameter.

    The resulting callback URL must either:
      - Be on the same domain as the application_url, or
      - Have a hostname included in ALLOWED_CALLBACK_DOMAINS.

    Args:
        application_url: Root URL for the application.
        callback_route: Endpoint for the OAuth callback handler (absolute URL or path).
        token_state: Unique identifier to maintain the integrity of the OAuth exchange.

    Returns:
        A fully assembled OAuth callback URL as a string.

    Raises:
        ValueError: If the final callback URL's domain is not the same as application_url's domain
                    and is not included in ALLOWED_CALLBACK_DOMAINS, or if the domain cannot be
                    determined from the inputs.
    """
    if not isinstance(application_url, str) or not isinstance(callback_route, str) or not isinstance(token_state, str):
        raise TypeError("application_url, callback_route, and token_state must all be strings")

    app_parsed = urllib.parse.urlparse(application_url)

    # Determine the full callback URL (absolute)
    route_parsed = urllib.parse.urlparse(callback_route)
    if route_parsed.scheme or route_parsed.netloc:
        # Absolute URL provided as callback_route
        full_callback_parsed = route_parsed
    else:
        # Relative path; resolve against the application_url
        if not app_parsed.scheme or not app_parsed.netloc:
            raise ValueError("application_url must be an absolute URL when callback_route is relative")
        full_callback_parsed = urllib.parse.urlparse(urllib.parse.urljoin(application_url, callback_route))

    # Validate domain restrictions (same-domain or allowed list)
    callback_host = (full_callback_parsed.hostname or "").lower()
    app_host = (app_parsed.hostname or "").lower()

    if not callback_host:
        raise ValueError("Unable to determine callback URL domain")

    if callback_host != app_host and callback_host not in {d.lower() for d in ALLOWED_CALLBACK_DOMAINS}:
        raise ValueError(f"OAuth callback domain not allowed: {callback_host}")

    # Append or replace the 'state' query parameter
    existing_qs = urllib.parse.parse_qsl(full_callback_parsed.query, keep_blank_values=True)
    updated_qs = []
    state_replaced = False
    for k, v in existing_qs:
        if k == "state":
            updated_qs.append(("state", token_state))
            state_replaced = True
        else:
            updated_qs.append((k, v))
    if not state_replaced:
        updated_qs.append(("state", token_state))

    new_query = urllib.parse.urlencode(updated_qs, doseq=True)
    final_url = urllib.parse.urlunparse(full_callback_parsed._replace(query=new_query))
    return final_url
