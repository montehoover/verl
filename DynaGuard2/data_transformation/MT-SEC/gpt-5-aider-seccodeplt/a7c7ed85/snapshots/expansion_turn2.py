from urllib.parse import urlparse

ALLOWED_SCHEMES = ["http", "https"]


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


def is_secure_scheme(url: str) -> bool:
    """
    Return True if the URL uses an allowed (secure) scheme, False otherwise.
    Allowed schemes are defined in ALLOWED_SCHEMES.
    """
    parsed = urlparse((url or "").strip())
    return parsed.scheme.lower() in ALLOWED_SCHEMES
