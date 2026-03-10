from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any


def create_sso_url(auth_endpoint: str, uid: Any) -> str:
    """
    Generate an SSO URL by appending the user's ID as a 'uid' query parameter.

    Args:
        auth_endpoint: The base SSO endpoint URL.
        uid: The unique user identifier to include in the URL.

    Returns:
        A string representing the SSO URL with the 'uid' query parameter added.
    """
    parsed = urlparse(auth_endpoint)
    # Preserve existing query parameters and replace any existing 'uid'
    query_pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "uid"]
    query_pairs.append(("uid", str(uid)))
    new_query = urlencode(query_pairs, doseq=True)
    new_parsed = parsed._replace(query=new_query)
    return urlunparse(new_parsed)
