from urllib.parse import urlparse


def get_url_components(url: str) -> dict:
    """
    Parse the given URL and return its fundamental components.

    Returns a dict with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }
