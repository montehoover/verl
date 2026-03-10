from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def construct_redirect_link(domain_base_url: str, next_redirect_param: str) -> str:
    """
    Construct a safe redirect URL for post-login redirection.

    Args:
        domain_base_url: Absolute base URL of the application (e.g., "https://app.example.com/login").
        next_redirect_param: The path or URL the user should be sent to after login.

    Returns:
        A string with domain_base_url augmented with a `next` query parameter.

    Raises:
        ValueError: If domain_base_url is not absolute, if next_redirect_param uses a disallowed scheme,
                    or if next_redirect_param points to a host neither within the base host nor in ALLOWED_DOMAINS.
    """
    if not isinstance(domain_base_url, str) or not domain_base_url.strip():
        raise ValueError("domain_base_url must be a non-empty string")

    base_parts = urlparse.urlsplit(domain_base_url)
    if not base_parts.scheme or not base_parts.netloc:
        raise ValueError("domain_base_url must be an absolute URL with scheme and host")

    base_host = base_parts.hostname or ""

    # Normalize the next parameter to a string (allow empty string).
    next_value = "" if next_redirect_param is None else str(next_redirect_param)
    next_parts = urlparse.urlsplit(next_value)

    # Disallow dangerous or non-HTTP(S) schemes in the `next` target
    if next_parts.scheme and next_parts.scheme.lower() not in ("http", "https"):
        raise ValueError("next_redirect_param uses a disallowed URL scheme")

    # Validate where `next` points to, if it specifies a host
    # Cases:
    # - Relative path (no scheme and no netloc): implicitly within base domain -> allowed
    # - Protocol-relative or absolute URL: validate host
    target_host = next_parts.hostname  # None for relative paths
    if target_host:
        # If target host is not the same as the application's host,
        # ensure it's in the allowed external domains list (supports subdomains).
        if not _is_same_host(target_host, base_host) and not _is_in_allowed_domains(target_host):
            raise ValueError("Redirection target is not within the base domain or allowed external domains")

    # Build the resulting URL with an updated `next` query parameter
    existing_qs = urlparse.parse_qsl(base_parts.query, keep_blank_values=True)
    # Remove existing 'next' parameters if present
    existing_qs = [(k, v) for (k, v) in existing_qs if k.lower() != "next"]
    existing_qs.append(("next", next_value))

    new_query = urlparse.urlencode(existing_qs, doseq=True)
    result = urlparse.urlunsplit((
        base_parts.scheme,
        base_parts.netloc,
        base_parts.path,
        new_query,
        base_parts.fragment,
    ))
    return result


def _is_same_host(candidate: str, base_host: str) -> bool:
    """
    Returns True if candidate host exactly matches the base host.
    """
    return (candidate or "").lower() == (base_host or "").lower()


def _is_in_allowed_domains(host: str) -> bool:
    """
    Returns True if host matches one of the allowed domains or their subdomains.
    """
    if not host:
        return False
    host = host.lower().rstrip(".")
    for allowed in ALLOWED_DOMAINS:
        allowed = allowed.lower().rstrip(".")
        if host == allowed or host.endswith("." + allowed):
            return True
    return False
