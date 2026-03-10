from urllib import parse as urlparse

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _normalize_target(target_param: str) -> str:
    """Trim whitespace and coerce None-like values to an empty string."""
    return (target_param or "").strip()


def _parse_absolute_http_url(url: str) -> urlparse.ParseResult:
    """
    Parse and validate that the provided URL is an absolute HTTP(S) URL.
    :raises ValueError: if invalid.
    """
    parsed = urlparse.urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        raise ValueError("Invalid main_url; must be an absolute HTTP(S) URL")
    return parsed


def _join_url(base_url: str, target: str) -> str:
    """
    Construct the final URL by resolving `target` against `base_url`
    using RFC-compliant resolution rules.
    """
    return urlparse.urljoin(base_url, target)


def _is_permitted_host(final_host: str, base_host: str, permitted_domains: list[str]) -> bool:
    """
    Determine if final_host is permitted:
    - same as base_host, or
    - matches any domain in permitted_domains (exact or subdomain).
    """
    if not final_host:
        return False

    fh = final_host.lower()
    bh = (base_host or "").lower()

    if fh == bh:
        return True

    for domain in permitted_domains:
        d = domain.lower().lstrip(".")
        if fh == d or fh.endswith("." + d):
            return True
    return False


def _validate_final_url_and_permissions(final_url: str, base_host: str, permitted_domains: list[str]) -> str:
    """
    Validate that final_url is a well-formed HTTP(S) URL and its host is permitted.
    :returns: normalized URL string
    :raises ValueError: when invalid or disallowed.
    """
    parsed_final = urlparse.urlparse(final_url)

    if parsed_final.scheme not in ("http", "https") or not parsed_final.hostname:
        raise ValueError("Disallowed redirect URL scheme or malformed URL")

    if not _is_permitted_host(parsed_final.hostname, base_host, permitted_domains):
        raise ValueError("Redirect URL domain is not permitted")

    return parsed_final.geturl()


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Constructs a safe redirect URL based on a base application URL and a target parameter.
    Ensures the resulting URL remains within the application's domain or an allowed domain.

    :param main_url: Base URL of the application.
    :param target_param: URL path or full URL to redirect to after sign-in.
    :return: Final redirect URL as a string.
    :raises ValueError: If the URL is invalid or points to a disallowed domain.
    """
    if not isinstance(main_url, str) or not isinstance(target_param, str):
        raise ValueError("main_url and target_param must be strings")

    base = _parse_absolute_http_url(main_url)
    target = _normalize_target(target_param)
    final_url = _join_url(main_url, target)
    return _validate_final_url_and_permissions(final_url, base.hostname, PERMITTED_DOMAINS)
