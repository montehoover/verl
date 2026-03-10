from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any


def create_sso_url(auth_endpoint: str, uid: Any, redirect_url: str) -> str:
    """
    Generate an SSO URL by appending the user's ID and a return URL as query parameters.

    Args:
        auth_endpoint: The base SSO endpoint URL.
        uid: The unique user identifier to include in the URL.
        redirect_url: The URL to redirect to after authentication. Must start with "https://".

    Returns:
        A string representing the SSO URL with the 'uid' and 'return_url' query parameters added.

    Raises:
        ValueError: If redirect_url does not start with "https://".
    """
    if not redirect_url.startswith("https://"):
        raise ValueError("Insecure redirect_url: must start with 'https://'")

    parsed = urlparse(auth_endpoint)
    # Preserve existing query parameters and replace any existing 'uid' or 'return_url'
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k not in {"uid", "return_url"}
    ]
    query_pairs.append(("uid", str(uid)))
    query_pairs.append(("return_url", redirect_url))
    new_query = urlencode(query_pairs, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)
