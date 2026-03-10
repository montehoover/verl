from urllib.parse import urlparse, urljoin, urlencode, urlunparse
from typing import Dict, Any, Iterable, Tuple


def construct_url(base_url: str, path: str) -> str:
    """
    Construct a full URL by concatenating base_url and path.

    Parameters:
        base_url (str): The base URL.
        path (str): The URL path.

    Returns:
        str: The concatenated URL.
    """
    return base_url + path


def _is_valid_domain(host: str) -> bool:
    """
    Validate that the provided host is a valid domain (e.g., example.com, sub.example.org)
    or 'localhost'. IP addresses and IPv6 literals are not considered valid domains here.
    """
    if not host:
        return False

    # Allow localhost explicitly
    if host.lower() == "localhost":
        return True

    # Convert possible Unicode domain to IDNA for validation
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except Exception:
        return False

    # Total length must be <= 253 chars
    if len(ascii_host) > 253:
        return False

    # Must contain at least one dot (e.g., example.com)
    if "." not in ascii_host:
        return False

    labels = ascii_host.split(".")
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        # Labels must be alphanumeric or hyphen, cannot start or end with hyphen
        if label.startswith("-") or label.endswith("-"):
            return False
        for ch in label:
            if not (ch.isalnum() or ch == "-"):
                return False

    # TLD should be at least 2 chars
    if len(labels[-1]) < 2:
        return False

    return True


def _normalize_query_params(query_params: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    """
    Normalize query params into a sequence of (key, value) pairs suitable for urlencode with doseq=True.
    - Skips keys with value None.
    - Expands list/tuple values.
    - Casts values to str.
    """
    for key, value in query_params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                yield (str(key), str(item))
        else:
            yield (str(key), str(value))


def construct_url_with_params(base_url: str, path: str, query_params: Dict[str, Any] | None = None) -> str:
    """
    Construct a full URL by combining base_url and path, and appending encoded query parameters.
    Ensures base_url uses http/https and has a valid domain.

    Parameters:
        base_url (str): The base URL (must include scheme and a valid domain).
        path (str): The URL path to append.
        query_params (dict | None): Dictionary of query parameters to encode and append.

    Returns:
        str: The complete URL.
    """
    parsed_base = urlparse(base_url)
    if parsed_base.scheme not in ("http", "https"):
        raise ValueError("base_url must use http or https scheme.")
    if not parsed_base.netloc:
        raise ValueError("base_url must include a network location (domain).")

    # Strip userinfo and port to extract host for validation
    netloc_no_userinfo = parsed_base.netloc.split("@")[-1]
    # Reject IPv6 literals [::1]
    if netloc_no_userinfo.startswith("[") and netloc_no_userinfo.endswith("]"):
        raise ValueError("base_url must include a valid domain, not an IPv6 literal.")
    host = netloc_no_userinfo.rsplit(":", 1)[0]

    if not _is_valid_domain(host):
        raise ValueError("base_url must include a valid domain.")

    # Ensure base_url ends with slash for predictable joining and strip leading slash from path
    base_for_join = base_url if base_url.endswith("/") else base_url + "/"
    joined = urljoin(base_for_join, path.lstrip("/"))

    if query_params:
        normalized_pairs = list(_normalize_query_params(query_params))
        if normalized_pairs:
            query_string = urlencode(normalized_pairs, doseq=True)
            parsed_joined = urlparse(joined)
            new_query = query_string if not parsed_joined.query else f"{parsed_joined.query}&{query_string}"
            joined = urlunparse(parsed_joined._replace(query=new_query))

    return joined
