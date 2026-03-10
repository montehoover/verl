import urllib.parse

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def build_user_url(base_url: str, path: str, user_id: str) -> str:
    """
    Build a user-specific URL by concatenating the base_url and path, then
    appending the user_id as a query parameter.

    Note: This function does not modify slashes between base_url and path.
    """
    url = f"{base_url}{path}"
    separator = '&' if '?' in url else '?'
    return f"{url}{separator}user_id={user_id}"


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Construct an SSO redirect URL by appending uid and redirect_url as query parameters.
    Validates that redirect_url's domain is included in ALLOWED_RETURN_DOMAINS.
    """
    parsed_redirect = urllib.parse.urlparse(redirect_url)
    hostname = parsed_redirect.hostname
    if not hostname or hostname not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("redirect_url domain is not approved")

    parsed_sso = urllib.parse.urlparse(sso_path)
    query = urllib.parse.parse_qs(parsed_sso.query, keep_blank_values=True)
    query['uid'] = [uid]
    query['redirect_url'] = [redirect_url]
    new_query = urllib.parse.urlencode(query, doseq=True)
    new_parsed_sso = parsed_sso._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed_sso)
