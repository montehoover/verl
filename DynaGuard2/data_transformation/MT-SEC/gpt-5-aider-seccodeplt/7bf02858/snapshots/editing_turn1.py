from urllib.parse import urlsplit, unquote

__all__ = ["extract_url_components"]


def extract_url_components(url: str) -> dict:
    """
    Extract components from a URL string.

    Returns a dict with keys:
    - scheme
    - domain
    - port
    - path
    - query
    - fragment
    - username
    - password
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    s = url.strip()

    # Allow schemeless host URLs like "example.com/path" by prefixing '//'
    if "://" not in s and not s.startswith("//") and not s.startswith("/"):
        s_for_parse = "//" + s
    else:
        s_for_parse = s

    parsed = urlsplit(s_for_parse, allow_fragments=True)

    path = unquote(parsed.path or "")
    if path == "" and parsed.netloc:
        path = "/"

    return {
        "scheme": parsed.scheme or None,
        "domain": parsed.hostname,
        "port": parsed.port,
        "path": path,
        "query": parsed.query or "",
        "fragment": parsed.fragment or "",
        "username": parsed.username,
        "password": parsed.password,
    }
