import urllib.parse
from typing import Any

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def create_sso_url(auth_endpoint: str, uid: Any, redirect_url: str) -> str:
    """
    Generate an SSO URL by appending the user's ID and a return URL as query parameters.
    Enforces that redirect_url starts with 'https://' and that its domain is in ALLOWED_RETURN_DOMAINS.

    Args:
        auth_endpoint: The base SSO endpoint URL.
        uid: The unique user identifier to include in the URL.
        redirect_url: The URL to redirect to after authentication. Must start with "https://"
                      and have a domain in ALLOWED_RETURN_DOMAINS.

    Returns:
        A string representing the SSO URL with the 'uid' and 'return_url' query parameters added.

    Raises:
        ValueError: If redirect_url does not start with "https://".
        ValueError: If redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure redirect_url: must start with 'https://'")

    redirect_parsed = urllib.parse.urlparse(redirect_url)
    redirect_host = redirect_parsed.hostname
    if not redirect_host or redirect_host not in ALLOWED_RETURN_DOMAINS:
        raise ValueError("Disallowed redirect_url domain")

    parsed = urllib.parse.urlparse(auth_endpoint)
    # Preserve existing query parameters and replace any existing 'uid' or 'return_url'
    query_pairs = [
        (k, v)
        for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if k not in {"uid", "return_url"}
    ]
    query_pairs.append(("uid", str(uid)))
    query_pairs.append(("return_url", redirect_url))
    new_query = urllib.parse.urlencode(query_pairs, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(new_parsed)
