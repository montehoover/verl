from urllib.parse import urlparse, urlunparse, urljoin, urlencode

ALLOWED_DOMAINS = ['example.com', 'trusted.com', 'secure.org']
_ALLOWED_SET = {d.lower().rstrip('.') for d in ALLOWED_DOMAINS}


def construct_and_validate_url(base_url: str, path: str, query_params: dict) -> str:
    """
    Construct a URL from base_url, path, and query_params, validate the domain against the allowed list,
    and return the full URL string. Raise ValueError if the domain is not allowed or the base URL is invalid.
    """
    if not isinstance(base_url, str) or not isinstance(path, str):
        raise TypeError("base_url and path must be strings")
    if not isinstance(query_params, dict):
        raise TypeError("query_params must be a dict")

    b = base_url.strip()
    p = path.strip()

    if not b:
        raise ValueError("base_url cannot be empty")

    # Ensure parsable URL: add a default scheme if missing (e.g., "example.com")
    if '://' not in b and not b.startswith('//'):
        b = 'http://' + b

    parsed_base = urlparse(b)
    host = (parsed_base.hostname or '').rstrip('.').lower()
    if not host:
        raise ValueError("Invalid base_url: missing hostname")

    # Validate domain against allowed list (exact match)
    if host not in _ALLOWED_SET:
        raise ValueError("Domain not allowed")

    # Build base without params/query/fragment
    base_no_qf = parsed_base._replace(params='', query='', fragment='').geturl()

    # Join path to base, ensuring a single slash boundary
    base_for_join = base_no_qf if base_no_qf.endswith('/') or p.startswith('/') else base_no_qf + '/'
    combined = urljoin(base_for_join, p)

    # Build query string; exclude items with value None
    qp_items = []
    for k, v in query_params.items():
        if v is None:
            continue
        qp_items.append((str(k), v))
    query_str = urlencode(qp_items, doseq=True)

    parsed_combined = urlparse(combined)
    final_url = urlunparse((
        parsed_combined.scheme,
        parsed_combined.netloc,
        parsed_combined.path,
        '',  # params
        query_str,
        ''   # fragment
    ))

    return final_url
