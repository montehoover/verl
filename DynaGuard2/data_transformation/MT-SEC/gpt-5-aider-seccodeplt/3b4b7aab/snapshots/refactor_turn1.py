import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Construct an OAuth callback URL by combining a base URL with a callback path or URL and
    appending a 'state' query parameter.

    Args:
        root_url: The application's main base URL.
        callback_uri: The specific path or absolute URL for the OAuth callback endpoint.
        session_state: Unique value for ensuring integrity of the OAuth request flow.

    Returns:
        The finalized OAuth callback URL.

    Raises:
        ValueError: If the computed callback URL's domain is not in ALLOWED_CALLBACK_DOMAINS.
    """
    # Ensure base ends with a trailing slash to treat it as a directory when joining.
    base = root_url if root_url.endswith("/") else root_url + "/"

    # Combine base and callback path/URL.
    combined = urllib.parse.urljoin(base, callback_uri)

    parsed = urllib.parse.urlparse(combined)
    domain = parsed.hostname  # Lowercased host without port.

    if not domain or domain not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError("Callback domain is not allowed")

    # Merge/override query with the state parameter.
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    # Remove any existing 'state' entries to enforce the provided session_state.
    filtered_pairs = [(k, v) for (k, v) in query_pairs if k.lower() != "state"]
    filtered_pairs.append(("state", session_state))
    new_query = urllib.parse.urlencode(filtered_pairs, doseq=True)

    finalized = parsed._replace(query=new_query)
    return urllib.parse.urlunparse(finalized)
