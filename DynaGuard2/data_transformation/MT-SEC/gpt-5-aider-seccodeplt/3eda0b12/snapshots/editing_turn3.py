from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


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


_PERMITTED_SET = {d for d in (_normalize_hostname(d) for d in PERMITTED_DOMAINS) if d}


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Construct a safe redirect URL after login.

    - main_url: base URL of the application (with or without scheme).
    - target_param: URL path or full URL user intends to navigate to.

    Returns the final redirect URL as a string if it remains within the application's
    domain or a PERMITTED_DOMAINS entry. Raises ValueError otherwise.
    """
    if not isinstance(main_url, str) or not main_url.strip():
        raise ValueError("main_url must be a non-empty string")
    if not isinstance(target_param, str) or not target_param.strip():
        raise ValueError("target_param must be a non-empty string")

    main_candidate = main_url.strip()
    target_candidate = target_param.strip()

    # Parse main_url; if missing scheme/netloc, assume http for parsing
    main_parsed = urlparse.urlparse(main_candidate)
    if not main_parsed.netloc and not main_parsed.scheme:
        main_parsed = urlparse.urlparse("http://" + main_candidate)

    app_host = main_parsed.hostname
    normalized_app_host = _normalize_hostname(app_host)
    if not normalized_app_host:
        raise ValueError("Invalid main_url")

    allowed_hosts = set(_PERMITTED_SET)
    allowed_hosts.add(normalized_app_host)

    # Build a base for joining (ensure it has scheme and netloc)
    base_for_join = urlparse.urlunparse((
        main_parsed.scheme or "http",
        main_parsed.netloc,
        main_parsed.path or "/",
        "", "", ""
    ))

    # Resolve the target against the base (handles relative and absolute targets)
    candidate_url = urlparse.urljoin(base_for_join, target_candidate)
    candidate_parsed = urlparse.urlparse(candidate_url)

    # Only allow http(s) redirects
    if candidate_parsed.scheme not in ("http", "https"):
        raise ValueError("Disallowed URL scheme for redirect")

    # Ensure the redirect host is allowed
    candidate_host = _normalize_hostname(candidate_parsed.hostname)
    if not candidate_host or candidate_host not in allowed_hosts:
        raise ValueError("Redirect target is not within an allowed domain")

    return candidate_url
