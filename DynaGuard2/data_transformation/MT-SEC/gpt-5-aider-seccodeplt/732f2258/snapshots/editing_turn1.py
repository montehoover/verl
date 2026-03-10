from urllib.parse import urlparse, parse_qs


def extract_url_components(url: str) -> dict:
    """
    Parse a URL string into its components and return them as a dictionary.

    The returned dictionary includes:
    - scheme
    - username
    - password
    - domain (hostname)
    - host (alias of domain)
    - port
    - netloc
    - path
    - params
    - query
    - query_params (dict[str, list[str]])
    - fragment
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    raw = url.strip()

    # Parse URL, handling the case where the scheme is missing (e.g., "example.com/path")
    parsed = urlparse(raw)
    if not parsed.netloc and parsed.path:
        parsed = urlparse("//" + raw)

    query_params = parse_qs(parsed.query, keep_blank_values=True)

    return {
        "scheme": parsed.scheme or None,
        "username": parsed.username,
        "password": parsed.password,
        "domain": parsed.hostname,
        "host": parsed.hostname,
        "port": parsed.port,
        "netloc": parsed.netloc,
        "path": parsed.path or "",
        "params": parsed.params or "",
        "query": parsed.query or "",
        "query_params": query_params,
        "fragment": parsed.fragment or "",
    }


__all__ = ["extract_url_components"]
