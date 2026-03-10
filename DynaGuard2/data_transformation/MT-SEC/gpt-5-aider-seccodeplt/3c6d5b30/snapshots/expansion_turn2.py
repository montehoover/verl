import re
import ipaddress
from urllib.parse import urlsplit, urlunsplit, urlencode, quote, parse_qsl


__all__ = ["validate_url", "concatenate_url_path"]


def _is_valid_domain(host: str) -> bool:
    """
    Validate a domain name according to common DNS rules:
    - Total length <= 253 (in ASCII/IDNA)
    - Each label 1..63 chars, starts/ends with alphanumeric, internal hyphens allowed
    - At least two labels (e.g., example.com)
    - TLD not all-numeric
    """
    try:
        ascii_host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return False

    if ascii_host.endswith("."):
        ascii_host = ascii_host[:-1]

    if not ascii_host or len(ascii_host) > 253:
        return False

    labels = ascii_host.split(".")
    if len(labels) < 2:
        return False

    label_re = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
    for label in labels:
        if not label or len(label) > 63 or not label_re.fullmatch(label):
            return False

    # TLD cannot be all numeric
    if labels[-1].isdigit():
        return False

    return True


def validate_url(url: str) -> bool:
    """
    Validate a URL intended for web use.
    - Requires http or https scheme
    - Requires a valid host (domain, IP, or 'localhost')
    - Optional port must be 1..65535
    Returns True if valid; raises ValueError if invalid.
    """
    if not isinstance(url, str):
        raise ValueError("URL must be a string")

    url = url.strip()
    if not url:
        raise ValueError("URL cannot be empty")

    parts = urlsplit(url)

    if parts.scheme not in ("http", "https"):
        raise ValueError("URL must start with http:// or https://")

    if not parts.netloc:
        raise ValueError("URL must include a host")

    host = parts.hostname
    if not host:
        raise ValueError("URL host is invalid")

    # Validate port if present
    try:
        port = parts.port
    except ValueError:
        raise ValueError("URL port is invalid or out of range")

    if port is not None and not (1 <= port <= 65535):
        raise ValueError("URL port out of range (1-65535)")

    host_norm = host.strip().lower().rstrip(".")

    # Allow localhost
    if host_norm == "localhost":
        return True

    # IP address (IPv4/IPv6)
    try:
        ipaddress.ip_address(host_norm)
        return True
    except ValueError:
        pass

    # Domain name
    if not _is_valid_domain(host_norm):
        raise ValueError("URL host is not a valid domain or IP address")

    return True


def concatenate_url_path(base_url: str, path: str) -> str:
    """
    Concatenate a base URL with a relative path, handling URL encoding.

    - base_url must be a valid http/https URL.
    - path must be a relative URL path (may include query and fragment).
    - Existing encoding in base_url is preserved; path/query/fragment from `path` are percent-encoded.
    - If `path` starts with '/', it is treated as an absolute path from the host root.
    - If `path` is empty or whitespace, the base_url is returned unchanged.
    """
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not isinstance(path, str):
        raise ValueError("path must be a string")

    base_url = base_url.strip()
    if not base_url:
        raise ValueError("base_url cannot be empty")

    # Validate base URL
    validate_url(base_url)

    if path.strip() == "":
        return base_url

    base = urlsplit(base_url)
    p = urlsplit(path)

    # Disallow full URLs in `path` to avoid surprising overrides of base_url
    if p.scheme or p.netloc:
        raise ValueError("path must be a relative URL path, not a full URL")

    # Determine the new path
    base_path = base.path or "/"
    if p.path:
        if p.path.startswith("/"):
            # Absolute from root
            joined_path = "/" + p.path.lstrip("/")
        else:
            # Append to base path as directory
            if not base_path.endswith("/"):
                base_path = base_path + "/"
            joined_path = base_path + p.path
    else:
        # No new path component; keep base path
        joined_path = base_path

    # Percent-encode the resulting path; preserve already-encoded sequences with '%'
    encoded_path = quote(joined_path, safe="/:@-._~!$&'()*+,;=%")

    # Query handling: if `path` provides a query, use it (encoded); otherwise keep base query
    if p.query:
        query_params = parse_qsl(p.query, keep_blank_values=True)
        new_query = urlencode(query_params, doseq=True, quote_via=quote)
    else:
        new_query = base.query

    # Fragment handling: if `path` provides a fragment, use it (encoded); otherwise keep base fragment
    if p.fragment:
        new_fragment = quote(p.fragment, safe="/:@-._~!$&'()*+,;=%")
    else:
        new_fragment = base.fragment

    return urlunsplit((base.scheme, base.netloc, encoded_path, new_query, new_fragment))
