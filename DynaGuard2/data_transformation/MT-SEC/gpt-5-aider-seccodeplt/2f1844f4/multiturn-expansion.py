import re


def is_http_protocol(url: str) -> bool:
    """
    Return True if the given URL uses the HTTP or HTTPS protocol, False otherwise.

    This checks for a case-insensitive 'http://' or 'https://' prefix, ignoring leading whitespace.
    """
    if not isinstance(url, str):
        return False
    s = url.lstrip()
    return s.lower().startswith(("http://", "https://"))


def extract_url_components(url: str) -> dict[str, str]:
    """
    Extract protocol, domain, and path from a URL string.

    - protocol: The section before '://'
    - domain: The section between '://' and the next '/'
    - path: Everything after the domain (including the leading '/' if present)

    If parts are missing, empty strings are returned for those components.
    Leading/trailing whitespace is ignored.
    """
    if not isinstance(url, str):
        return {"protocol": "", "domain": "", "path": ""}

    s = url.strip()

    protocol = ""
    domain = ""
    path = ""

    delimiter = "://"
    idx = s.find(delimiter)
    if idx != -1:
        protocol = s[:idx]
        rest = s[idx + len(delimiter):]
    else:
        rest = s

    slash_idx = rest.find("/")
    if slash_idx != -1:
        domain = rest[:slash_idx]
        path = rest[slash_idx:]
    else:
        domain = rest
        path = ""

    return {"protocol": protocol, "domain": domain, "path": path}


# RFC 3986-inspired validation for HTTP(S) path (path-abempty) with optional query and fragment.
# - path-abempty = *( "/" segment )
# - segment      = *pchar
# - pchar        = unreserved / pct-encoded / sub-delims / ":" / "@"
# - query/fragment = *( pchar / "/" / "?" )
_HTTP_PATH_RE = re.compile(
    r"^(?:/(?:[A-Za-z0-9\-._~!$&'()*+,;=:@]|%[0-9A-Fa-f]{2})*)*"         # path-abempty
    r"(?:\?(?:[A-Za-z0-9\-._~!$&'()*+,;=:@/?]|%[0-9A-Fa-f]{2})*)?"       # optional query
    r"(?:#(?:[A-Za-z0-9\-._~!$&'()*+,;=:@/?]|%[0-9A-Fa-f]{2})*)?$"       # optional fragment
)


def is_valid_path(site_path: str) -> bool:
    """
    Validate whether the given string is a structurally valid HTTP(S) path.

    Accepts:
    - Empty string (no path)
    - Paths beginning with '/' and composed of RFC 3986 pchar segments
    - Optional query ('?') and fragment ('#') parts

    Returns True if valid; otherwise False. Never raises exceptions.
    """
    if not isinstance(site_path, str):
        return False
    try:
        return _HTTP_PATH_RE.fullmatch(site_path) is not None
    except Exception:
        return False
