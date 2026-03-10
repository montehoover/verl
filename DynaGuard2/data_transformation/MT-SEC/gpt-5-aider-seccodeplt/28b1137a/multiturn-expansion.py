import urllib.parse
from urllib.parse import urlsplit, urlencode
from collections.abc import Mapping

# Predefined list of approved domains (edit as needed)
APPROVED_DOMAINS = {
    "example.com",
    "api.example.com",
}

# Allowed return domains for SSO redirect target validation
ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _normalize_domain(domain: str) -> str:
    """
    Normalize a domain for consistent comparison:
    - strip whitespace
    - lowercase
    - remove trailing dot
    - convert to IDNA (punycode) ASCII representation
    """
    if not isinstance(domain, str):
        raise TypeError("domain must be a string")
    s = domain.strip().lower().rstrip(".")
    if not s:
        raise ValueError("Empty domain")
    try:
        return s.encode("idna").decode("ascii")
    except Exception as exc:
        raise ValueError(f"Invalid domain name: {domain}") from exc


# Normalized sets for efficient membership checks
APPROVED_DOMAINS_NORMALIZED = {_normalize_domain(d) for d in APPROVED_DOMAINS}
ALLOWED_RETURN_DOMAINS_NORMALIZED = {_normalize_domain(d) for d in ALLOWED_RETURN_DOMAINS}


def validate_domain(url: str) -> bool:
    """
    Validate that the URL's domain is in the approved list.

    Returns:
        True if the domain is approved.

    Raises:
        ValueError: If the URL is empty/invalid or the domain is not approved.
        TypeError: If url is not a string.
    """
    if not isinstance(url, str):
        raise TypeError("url must be a string")

    candidate = url.strip()
    if not candidate:
        raise ValueError("Empty URL")

    # Allow URLs without scheme by assuming http
    parsed = urlsplit(candidate if "://" in candidate else f"http://{candidate}")
    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a host")

    normalized_host = _normalize_domain(host)

    if normalized_host in APPROVED_DOMAINS_NORMALIZED:
        return True

    raise ValueError(f"Unapproved domain: {host}")


def create_query_string(params: dict) -> str:
    """
    Create a URL-encoded query string from a dictionary of parameters.

    - Keys are converted to strings.
    - Values:
        * None values are omitted.
        * Booleans are serialized as "true"/"false".
        * Other values are converted to strings.
        * Lists/tuples expand into repeated keys (e.g., k=v1&k=v2).

    Returns:
        A URL-encoded query string without a leading '?'.
    """
    if not isinstance(params, Mapping):
        raise TypeError("params must be a mapping (e.g., dict)")

    def _convert_value(v):
        if v is None:
            return None
        if isinstance(v, bool):
            return "true" if v else "false"
        return str(v)

    pairs = []
    for k, v in params.items():
        if v is None:
            continue
        key = str(k)
        if isinstance(v, (list, tuple)):
            converted = [_convert_value(x) for x in v]
            for item in converted:
                if item is not None:
                    pairs.append((key, item))
        else:
            converted = _convert_value(v)
            if converted is not None:
                pairs.append((key, converted))

    return urlencode(pairs, doseq=True)


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Build a complete SSO redirect URL.

    Args:
        sso_path: Base SSO URL or path (may already contain a query string).
        uid: User identifier to include in the SSO URL.
        redirect_url: URL to return to after SSO. Its domain must be allowed.

    Returns:
        A full SSO redirect URL as a string.

    Raises:
        ValueError: If any required argument is empty or redirect_url's domain is not allowed.
        TypeError: If arguments are not strings.
    """
    # Type and value checks
    for name, value in (("sso_path", sso_path), ("uid", uid), ("redirect_url", redirect_url)):
        if not isinstance(value, str):
            raise TypeError(f"{name} must be a string")
        if not value.strip():
            raise ValueError(f"{name} cannot be empty")

    # Validate redirect_url against allowed return domains
    redir_candidate = redirect_url.strip()
    redir_parsed = urllib.parse.urlsplit(redir_candidate if "://" in redir_candidate else f"http://{redir_candidate}")
    redir_host = redir_parsed.hostname
    if not redir_host:
        raise ValueError("redirect_url must include a host")
    redir_host_norm = _normalize_domain(redir_host)
    if redir_host_norm not in ALLOWED_RETURN_DOMAINS_NORMALIZED:
        raise ValueError(f"redirect_url domain not allowed: {redir_host}")

    # Build final SSO URL by merging existing query parameters with ours
    sso_parsed = urllib.parse.urlsplit(sso_path)
    existing_params = urllib.parse.parse_qsl(sso_parsed.query, keep_blank_values=True)
    merged_params = existing_params + [("uid", uid), ("redirect_url", redirect_url)]
    new_query = urllib.parse.urlencode(merged_params, doseq=True)

    return urllib.parse.urlunsplit((
        sso_parsed.scheme,
        sso_parsed.netloc,
        sso_parsed.path,
        new_query,
        sso_parsed.fragment,
    ))
