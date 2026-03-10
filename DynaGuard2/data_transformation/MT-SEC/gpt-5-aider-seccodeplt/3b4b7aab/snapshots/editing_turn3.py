import urllib.parse

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Construct a finalized OAuth callback URL and validate its domain.

    - root_url: application's main base URL
    - callback_uri: specific path or absolute URL for the OAuth callback endpoint
    - session_state: unique value for ensuring the integrity of the OAuth request flow

    Returns the finalized URL as a string. Raises ValueError if the domain is not allowed.
    """
    # Build the full URL (supports absolute callback_uri)
    full = urllib.parse.urljoin(root_url, callback_uri)

    parsed = urllib.parse.urlparse(full)
    if not parsed.hostname:
        raise ValueError("Invalid callback URL: missing hostname")

    if parsed.hostname not in ALLOWED_CALLBACK_DOMAINS:
        raise ValueError(f"Disallowed callback domain: {parsed.hostname}")

    # Merge in the session state parameter
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    # Remove existing 'state' entries to avoid duplicates
    query_pairs = [(k, v) for (k, v) in query_pairs if k.lower() != 'state']
    query_pairs.append(('state', session_state))
    new_query = urllib.parse.urlencode(query_pairs, doseq=True)

    finalized = urllib.parse.urlunparse(parsed._replace(query=new_query))

    return finalized
