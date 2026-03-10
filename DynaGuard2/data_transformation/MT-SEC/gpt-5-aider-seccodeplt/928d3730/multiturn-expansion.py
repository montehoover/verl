import urllib.parse
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode, quote
import ipaddress
import re
from typing import Set, Mapping, Any

# Predefined set of allowed domains (edit this set to fit your allowlist)
ALLOWED_DOMAINS: Set[str] = {
    "example.com",
    "example.org",
}

# Allowed OAuth callback hostnames (exact matches)
ALLOWED_CALLBACK_DOMAINS: Set[str] = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}

_LABEL_RE = re.compile(r"^[a-z0-9-]+$", re.IGNORECASE)


def _canonicalize_hostname(host: str) -> str:
    """
    Convert a hostname to a canonical ASCII (IDNA) lowercase form without a trailing dot.
    Returns an empty string if host is falsy.
    """
    if not host:
        return ""
    h = host.strip().rstrip(".").lower()
    # Convert Unicode domains to ASCII punycode for consistent comparison
    try:
        h_ascii = h.encode("idna").decode("ascii")
    except Exception:
        # If IDNA encoding fails, keep the original; downstream validation may reject it
        h_ascii = h
    return h_ascii


def _is_valid_hostname_ascii(host: str) -> bool:
    """
    Basic validation for an ASCII hostname (not an IP).
    - Total length <= 253
    - Each label 1..63 chars, alnum or hyphen, not starting/ending with hyphen
    """
    if not host:
        return False
    if len(host) > 253:
        return False
    labels = host.split(".")
    for label in labels:
        if not label:
            return False
        if len(label) > 63:
            return False
        if label[0] == "-" or label[-1] == "-":
            return False
        if not _LABEL_RE.fullmatch(label):
            return False
    return True


# Canonicalize allowed domains up-front
_ALLOWED_DOMAINS_CANON: Set[str] = {_canonicalize_hostname(d) for d in ALLOWED_DOMAINS if d}
# Canonicalize allowed callback domains up-front
_ALLOWED_CALLBACKS_CANON: Set[str] = {_canonicalize_hostname(d) for d in ALLOWED_CALLBACK_DOMAINS if d}


def validate_url_domain(url: str) -> bool:
    """
    Validate a URL and check if its domain is in the allowed list.

    - Returns True if the URL is valid and its hostname matches an allowed domain
      (either exact match or a subdomain of an allowed domain).
    - Returns False if the URL is valid but its hostname does not match the allowlist.
    - Raises ValueError if the URL itself is invalid.

    Only http and https schemes are considered valid for this check.
    """
    if not isinstance(url, str):
        raise ValueError("Invalid URL")

    candidate = url.strip()
    if not candidate:
        raise ValueError("Invalid URL")

    try:
        parts = urlsplit(candidate)
    except Exception as e:
        raise ValueError("Invalid URL") from e

    if parts.scheme not in ("http", "https"):
        raise ValueError("Invalid URL")

    if not parts.netloc:
        raise ValueError("Invalid URL")

    # Validate port if present (accessing .port triggers validation in urllib.parse)
    try:
        _ = parts.port  # noqa: F841
    except Exception as e:
        raise ValueError("Invalid URL") from e

    host = parts.hostname
    if not host:
        raise ValueError("Invalid URL")

    # If host is an IP address, URL is valid but IPs are not domain-allowed
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass  # Not an IP, proceed with domain validation

    host_canon = _canonicalize_hostname(host)
    if not _is_valid_hostname_ascii(host_canon):
        raise ValueError("Invalid URL")

    # Match exact allowed domain or any of its subdomains
    for allowed in _ALLOWED_DOMAINS_CANON:
        if host_canon == allowed or host_canon.endswith("." + allowed):
            return True

    return False


def build_url_with_params(base_url: str, path: str, params: Mapping[str, Any]) -> str:
    """
    Build a complete URL by combining a base URL, a path, and query parameters.

    - base_url: Must include scheme (http/https) and netloc.
    - path: Path to append. If it starts with '/', it is treated as an absolute path
      (replacing the base path). Otherwise it is appended to the base path.
    - params: Mapping of query parameters. Values of None are omitted. Values may be lists/tuples.

    Returns the fully constructed URL.
    Raises ValueError for invalid inputs or base_url.
    """
    if not isinstance(base_url, str) or not isinstance(path, str):
        raise ValueError("Invalid inputs")

    base = base_url.strip()
    if not base:
        raise ValueError("Invalid base URL")

    try:
        parts = urlsplit(base)
    except Exception as e:
        raise ValueError("Invalid base URL") from e

    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("Invalid base URL")

    # Validate port if present (accessing .port triggers validation in urllib.parse)
    try:
        _ = parts.port  # noqa: F841
    except Exception as e:
        raise ValueError("Invalid base URL") from e

    base_path = parts.path or ""
    add_path = path.strip()

    # Build combined path
    if add_path:
        if add_path.startswith("/"):
            combined_path = add_path
        else:
            if base_path and not base_path.endswith("/"):
                combined_path = base_path + "/" + add_path
            else:
                combined_path = (base_path or "/") + add_path
    else:
        combined_path = base_path or "/"

    # Normalize duplicate slashes and ensure leading slash
    combined_path = re.sub("/{2,}", "/", combined_path)
    if not combined_path.startswith("/"):
        combined_path = "/" + combined_path

    # Percent-encode the path safely (preserve '/' and existing escapes)
    encoded_path = quote(combined_path, safe="/%:@-._~!$&'()*+,;=")

    # Merge existing query parameters with provided ones (provided params override)
    existing_pairs = parse_qsl(parts.query, keep_blank_values=True)
    merged: dict[str, Any] = {}
    for k, v in existing_pairs:
        merged[k] = v

    if not isinstance(params, Mapping):
        raise ValueError("params must be a mapping")

    for k, v in params.items():
        if v is None:
            # Omit keys with None values
            merged.pop(k, None)
        else:
            merged[k] = v

    query = urlencode(merged, doseq=True)

    return urlunsplit((parts.scheme, parts.netloc, encoded_path, query, parts.fragment))


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Generate a secure OAuth callback URL.

    - host_url: Base HTTPS URL of the OAuth domain.
    - path_callback: Callback path to append.
    - session_id: Session identifier to embed as a 'session_id' query parameter.

    Validates that the callback host is in ALLOWED_CALLBACK_DOMAINS (exact match).
    Raises ValueError on invalid inputs or disallowed host.
    """
    if not isinstance(host_url, str) or not isinstance(path_callback, str) or not isinstance(session_id, str):
        raise ValueError("Invalid inputs")

    base = host_url.strip()
    if not base:
        raise ValueError("Invalid host_url")

    try:
        parts = urllib.parse.urlsplit(base)
    except Exception as e:
        raise ValueError("Invalid host_url") from e

    if parts.scheme != "https":
        raise ValueError("Invalid host_url")

    if not parts.netloc:
        raise ValueError("Invalid host_url")

    # Validate port if present (accessing .port triggers validation in urllib.parse)
    try:
        _ = parts.port  # noqa: F841
    except Exception as e:
        raise ValueError("Invalid host_url") from e

    host = parts.hostname
    if not host:
        raise ValueError("Invalid host_url")

    # Disallow IP addresses for callback hosts
    try:
        ipaddress.ip_address(host)
        raise ValueError("Callback domain not allowed")
    except ValueError:
        pass  # Not an IP; continue

    host_canon = _canonicalize_hostname(host)
    if not _is_valid_hostname_ascii(host_canon):
        raise ValueError("Invalid host_url")

    if host_canon not in _ALLOWED_CALLBACKS_CANON:
        raise ValueError("Callback domain not allowed")

    sid = session_id.strip()
    if not sid:
        raise ValueError("Invalid session_id")

    return build_url_with_params(base, path_callback, {"session_id": sid})
