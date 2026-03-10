from urllib.parse import urlparse


def parse_url(url: str) -> dict:
    """
    Parse a URL into its components and return them as a dictionary.

    Returns keys:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
    - username
    - password
    - hostname
    - port
    """
    if url is None:
        raise ValueError("url must be a non-empty string")
    if not isinstance(url, str):
        raise TypeError("url must be a string")
    url = url.strip()
    if not url:
        raise ValueError("url must be a non-empty string")

    result = urlparse(url)

    # Fallback: handle schemeless URLs like "example.com/path" by coercing netloc parsing.
    if not result.scheme and not result.netloc and '//' not in url:
        result = urlparse('//' + url)

    return {
        "scheme": result.scheme,
        "netloc": result.netloc,
        "path": result.path,
        "params": result.params,
        "query": result.query,
        "fragment": result.fragment,
        "username": result.username,
        "password": result.password,
        "hostname": result.hostname,
        "port": result.port,
    }
