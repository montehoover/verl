from urllib.parse import urlparse

def extract_url_parts(url: str):
    """
    Parse the given URL and return its components.

    Returns a dictionary with keys:
    - scheme
    - netloc
    - path
    - params
    - query
    - fragment
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
