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
