import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _extract_hostname(url: str):
    """
    Parse the URL and return a normalized (lowercase) hostname or None if not present.
    """
    parsed = urllib.parse.urlparse(url)
    return parsed.hostname.lower() if parsed.hostname else None


def _validate_redirect_domain(redirect_url: str, allowed_domains) -> None:
    """
    Ensure the redirect_url's domain is included in the allowed_domains set.
    Raises ValueError if invalid.
    """
    domain = _extract_hostname(redirect_url)
    if not domain or domain not in allowed_domains:
        raise ValueError("redirect_url domain is not allowed")


def _build_query_string(existing_query: str, updates: dict) -> str:
    """
    Merge existing query parameters with updates and return a new query string.
    """
    query_params = dict(urllib.parse.parse_qsl(existing_query, keep_blank_values=True))
    query_params.update(updates)
    return urllib.parse.urlencode(query_params, doseq=True)


def _rebuild_url_with_query(parsed_url: urllib.parse.ParseResult, query: str) -> str:
    """
    Return a URL string by replacing the query part of parsed_url with the provided query.
    """
    new_parsed = parsed_url._replace(query=query)
    return urllib.parse.urlunparse(new_parsed)


def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL by combining the SSO endpoint with the user ID and a validated return URL.

    Raises:
        ValueError: If redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    # Validate redirect URL domain
    _validate_redirect_domain(redirect_url, ALLOWED_RETURN_DOMAINS)

    # Parse the auth endpoint and prepare query parameters
    parsed_auth = urllib.parse.urlparse(auth_endpoint)
    new_query = _build_query_string(
        existing_query=parsed_auth.query,
        updates={"uid": uid, "redirect_url": redirect_url},
    )

    # Rebuild and return the final URL
    return _rebuild_url_with_query(parsed_auth, new_query)
