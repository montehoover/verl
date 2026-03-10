from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

APPROVED_DOMAINS = ['example.com', 'myapp.com', 'trusted.com']

# Pre-normalize approved domains for comparison
def _normalize_hostname(hostname: str) -> str | None:
    if not hostname:
        return None
    hn = hostname.strip().rstrip('.').lower()
    try:
        # Normalize internationalized domains to ASCII using IDNA
        hn_idna = hn.encode('idna').decode('ascii')
    except Exception:
        return None
    return hn_idna

_APPROVED_SET = {d for d in (_normalize_hostname(d) for d in APPROVED_DOMAINS) if d}


def construct_and_validate_url(base_url: str, params: dict) -> str:
    """
    Construct a URL from base_url and query params, validate the domain against
    APPROVED_DOMAINS, and return the complete URL if valid. Raise ValueError if not.

    - base_url: string URL that may or may not include a scheme and/or existing query.
    - params: dict of query parameters; values may be scalars or sequences.
    """
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be a non-empty string")
    if params is None:
        params = {}
    if not isinstance(params, dict):
        raise ValueError("params must be a dict of query parameters")

    candidate = base_url.strip()
    parsed = urlparse(candidate)

    # If no scheme and no netloc, try with http:// to parse the host
    if not parsed.netloc and not parsed.scheme:
        parsed = urlparse("http://" + candidate)

    host = parsed.hostname  # Strips port automatically, handles IPv6 brackets
    normalized_host = _normalize_hostname(host)

    if not normalized_host or normalized_host not in _APPROVED_SET:
        raise ValueError("Domain is not approved")

    # Merge existing query params with new params
    existing_params = parse_qsl(parsed.query, keep_blank_values=True)
    new_items = []
    for k, v in params.items():
        if v is None:
            continue
        new_items.append((str(k), v))  # urlencode with doseq=True will expand sequences

    full_query = urlencode(existing_params + new_items, doseq=True)

    rebuilt = parsed._replace(query=full_query)
    return urlunparse(rebuilt)
