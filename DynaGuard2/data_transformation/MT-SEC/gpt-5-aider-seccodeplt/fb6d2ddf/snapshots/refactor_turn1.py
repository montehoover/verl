import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}

def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL by combining the SSO endpoint with the user ID and a validated return URL.

    Raises:
        ValueError: If redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    domain = parsed_redirect.hostname.lower() if parsed_redirect.hostname else None
    if not domain or domain not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("redirect_url domain is not allowed")

    parsed_auth = urllib.parse.urlparse(auth_endpoint)
    query_params = dict(urllib.parse.parse_qsl(parsed_auth.query, keep_blank_values=True))
    query_params["uid"] = uid
    query_params["redirect_url"] = redirect_url

    new_query = urllib.parse.urlencode(query_params, doseq=True)
    new_auth = parsed_auth._replace(query=new_query)
    return urllib.parse.urlunparse(new_auth)
