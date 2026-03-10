from urllib.parse import urlparse

def extract_url_parts(url: str) -> dict:
    """
    Extract components of a URL string into a dictionary.

    Returns keys: scheme, netloc, path, params, query, fragment, username, password, hostname, port.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
        "username": parsed.username,
        "password": parsed.password,
        "hostname": parsed.hostname,
        "port": parsed.port,
    }
