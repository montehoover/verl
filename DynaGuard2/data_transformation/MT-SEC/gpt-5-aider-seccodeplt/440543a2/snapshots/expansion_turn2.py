from urllib.parse import urlparse, urlencode
from collections.abc import Mapping

NETWORK_SCHEMES = {"http", "https", "ftp", "ftps", "ws", "wss"}


def parse_and_validate_url(url: str) -> dict:
    """
    Parse and validate a URL string.

    Returns a dictionary with components: scheme, netloc, path, params, query, fragment.
    Raises ValueError if the URL is invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")
    url = url.strip()
    if not url:
        raise ValueError("URL must be a non-empty string")

    parsed = urlparse(url)

    scheme = parsed.scheme
    if not scheme:
        raise ValueError("URL is missing a scheme")

    scheme_lower = scheme.lower()

    if scheme_lower in NETWORK_SCHEMES:
        if not parsed.netloc:
            raise ValueError(f"URL is missing a network location (netloc) for scheme '{scheme_lower}'")
    elif scheme_lower == "file":
        # file URLs must include at least a path (or a host with a path)
        if not parsed.netloc and not parsed.path:
            raise ValueError("File URL must include a path")
    else:
        # For non-network, non-file schemes, require at least a path
        if not parsed.path:
            raise ValueError(f"URL must include a path for scheme '{scheme_lower}'")

    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def build_query_string(params: Mapping) -> str:
    """
    Build a URL-encoded query string from a dictionary-like mapping.

    - Skips parameters with value None.
    - Expands sequences (e.g., lists/tuples) into repeated keys (doseq=True).
    Returns an empty string if no parameters are provided or all values are None.
    """
    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping (e.g., dict)")

    filtered = {k: v for k, v in params.items() if v is not None}
    if not filtered:
        return ""

    return urlencode(filtered, doseq=True)
