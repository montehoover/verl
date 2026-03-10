from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']
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

    parsed_base = urlparse.urlparse(b)
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
    combined = urlparse.urljoin(base_for_join, p)

    # Build query string; exclude items with value None
    qp_items = []
    for k, v in query_params.items():
        if v is None:
            continue
        qp_items.append((str(k), v))
    query_str = urlparse.urlencode(qp_items, doseq=True)

    parsed_combined = urlparse.urlparse(combined)
    final_url = urlparse.urlunparse((
        parsed_combined.scheme,
        parsed_combined.netloc,
        parsed_combined.path,
        '',  # params
        query_str,
        ''   # fragment
    ))

    return final_url


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generate a safe redirect URL.
    - If redirect_param is a relative path, it will be resolved against app_base_url and must remain on the same host.
    - If redirect_param is an absolute URL (including scheme-relative //host), it must match an allowed external domain.
    Returns the final URL as a string, or raises ValueError if validation fails.
    """
    if not isinstance(app_base_url, str) or not isinstance(redirect_param, str):
        raise TypeError("app_base_url and redirect_param must be strings")

    base = app_base_url.strip()
    target = redirect_param.strip()

    if not base:
        raise ValueError("app_base_url cannot be empty")

    # Ensure base has a scheme for parsing/joining
    if '://' not in base and not base.startswith('//'):
        base = 'https://' + base

    parsed_base = urlparse.urlparse(base)
    base_host = (parsed_base.hostname or '').rstrip('.').lower()
    if not base_host:
        raise ValueError("Invalid app_base_url: missing hostname")

    # Normalize base URL (remove params/query/fragment)
    base_origin = parsed_base._replace(params='', query='', fragment='').geturl()

    # Default to root if empty
    if not target:
        target = '/'

    # Parse the target to determine if it is absolute
    parsed_target = urlparse.urlparse(target)

    def _is_allowed_external(hostname: str) -> bool:
        if not hostname:
            return False
        h = hostname.rstrip('.').lower()
        if h in _ALLOWED_SET:
            return True
        # Allow subdomains of allowed domains
        for allowed in _ALLOWED_SET:
            if h.endswith('.' + allowed):
                return True
        return False

    # If absolute (scheme or netloc present) or scheme-relative (starts with //)
    is_scheme_relative = target.startswith('//')
    is_absolute = bool(parsed_target.scheme or parsed_target.netloc or is_scheme_relative)

    if is_absolute:
        # Disallow non-http(s) schemes
        if parsed_target.scheme and parsed_target.scheme.lower() not in ('http', 'https'):
            raise ValueError("Unsupported redirect scheme")

        # Build a normalized absolute URL, filling scheme for //host with base scheme
        if is_scheme_relative:
            normalized_abs = urlparse.urlunparse((
                parsed_base.scheme,
                parsed_target.netloc,
                parsed_target.path,
                parsed_target.params,
                parsed_target.query,
                parsed_target.fragment
            ))
            parsed_abs = urlparse.urlparse(normalized_abs)
        else:
            parsed_abs = parsed_target

        # Disallow userinfo in netloc
        if parsed_abs.username or parsed_abs.password:
            raise ValueError("Userinfo not allowed in redirect URL")

        host = (parsed_abs.hostname or '').rstrip('.').lower()
        if not _is_allowed_external(host):
            raise ValueError("Redirect host not in allowed external domains")

        # Return the normalized absolute URL as the redirect target
        return urlparse.urlunparse((
            parsed_abs.scheme,
            parsed_abs.netloc,
            parsed_abs.path,
            parsed_abs.params,
            parsed_abs.query,
            parsed_abs.fragment
        ))

    # Relative path: resolve against base and ensure it stays on the same host
    base_for_join = base_origin if base_origin.endswith('/') or target.startswith('/') else base_origin + '/'
    combined = urlparse.urljoin(base_for_join, target)
    parsed_combined = urlparse.urlparse(combined)

    # Must stay within the same host as the app base
    combined_host = (parsed_combined.hostname or '').rstrip('.').lower()
    if combined_host != base_host:
        raise ValueError("Redirect escapes base domain")

    # Disallow userinfo in final URL
    if parsed_combined.username or parsed_combined.password:
        raise ValueError("Userinfo not allowed in redirect URL")

    return urlparse.urlunparse((
        parsed_combined.scheme,
        parsed_combined.netloc,
        parsed_combined.path,
        parsed_combined.params,
        parsed_combined.query,
        parsed_combined.fragment
    ))
