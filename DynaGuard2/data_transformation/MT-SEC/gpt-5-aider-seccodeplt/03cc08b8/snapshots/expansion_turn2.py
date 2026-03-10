from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote
import ipaddress
from typing import Set, Any, Dict, List, Tuple

# Predefined list of allowed domains/hosts.
# Update this set as needed for your application.
ALLOWED_DOMAINS = {
    "example.com",
    "api.example.com",
    "localhost",
    "127.0.0.1",
}

def _normalize_host_for_compare(host: str) -> str:
    """
    Normalize a hostname or IP for comparison:
    - For IP addresses (v4/v6), return the compressed canonical form.
    - For domain names, return lower-case ASCII using IDNA (punycode) and strip any trailing dot.
    """
    if not isinstance(host, str):
        return ""

    host = host.strip()
    if not host:
        return ""

    # Try IP normalization
    try:
        ip = ipaddress.ip_address(host)
        return ip.compressed
    except ValueError:
        pass

    # Domain normalization: remove trailing dot (FQDN) and convert to ASCII (IDNA), lower-case
    domain = host.rstrip(".")
    try:
        domain_ascii = domain.encode("idna").decode("ascii")
    except UnicodeError:
        # If IDNA encoding fails, fallback to lower-case as-is
        domain_ascii = domain
    return domain_ascii.lower()

# Pre-normalize allowed hosts for efficient comparison
_ALLOWED_HOSTS: Set[str] = {_normalize_host_for_compare(h) for h in ALLOWED_DOMAINS}

def validate_url_domain(url: str) -> bool:
    """
    Validate that the URL is well-formed and that its hostname is in the allowed list.

    Args:
        url: The URL string to validate.

    Returns:
        True if the URL's hostname is in the allowed list, otherwise False.

    Raises:
        ValueError: If the URL is invalid (e.g., missing hostname).
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL: must be a non-empty string")

    parts = urlsplit(url)

    # urlsplit ensures hostname is parsed only for URLs with a network location
    host = parts.hostname
    if not host:
        raise ValueError("Invalid URL: missing hostname")

    normalized_host = _normalize_host_for_compare(host)
    if not normalized_host:
        raise ValueError("Invalid URL: could not normalize hostname")

    return normalized_host in _ALLOWED_HOSTS

def build_url_with_params(base_url: str, path: str, params: Dict[str, Any]) -> str:
    """
    Build a complete URL by appending a path and query parameters to a base URL.

    Args:
        base_url: The base URL (including scheme and host).
        path: The path to append to the base URL's path.
        params: A dict of query parameters to add. Values may be scalars or lists/tuples.

    Returns:
        A fully constructed URL string.

    Raises:
        ValueError: If inputs are invalid or base_url lacks scheme/netloc.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be a dict")

    parts = urlsplit(base_url)
    if not parts.scheme or not parts.netloc:
        raise ValueError("Invalid base_url: must include scheme and network location")

    # Prepare the path: append provided path to the base path, preserving slashes.
    base_path = parts.path or ""
    base_path_no_trailing = base_path.rstrip("/")
    new_path = path.strip().lstrip("/")

    # Percent-encode path segments to be safe, while preserving segment separators.
    if new_path:
        encoded_segments = [quote(seg, safe="") for seg in new_path.split("/")]
        encoded_new_path = "/".join(encoded_segments)
    else:
        encoded_new_path = ""

    merged_path = "/".join(p for p in [base_path_no_trailing, encoded_new_path] if p)
    # Ensure leading slash for URLs with a netloc
    if parts.netloc and not merged_path.startswith("/"):
        merged_path = "/" + merged_path

    # Merge existing query params from base_url with provided params.
    # Provided params override any existing keys.
    existing_pairs: List[Tuple[str, str]] = parse_qsl(parts.query, keep_blank_values=True)
    keys_to_override = set(params.keys())
    filtered_pairs = [(k, v) for (k, v) in existing_pairs if k not in keys_to_override]

    new_pairs: List[Tuple[str, str]] = []
    for k, v in params.items():
        if v is None:
            # Skip None values
            continue
        if isinstance(v, (list, tuple)):
            for item in v:
                new_pairs.append((k, "" if item is None else str(item)))
        else:
            new_pairs.append((k, str(v)))

    combined_pairs = filtered_pairs + new_pairs
    query = urlencode(combined_pairs, doseq=True)

    return urlunsplit((parts.scheme, parts.netloc, merged_path, query, parts.fragment))
