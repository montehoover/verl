from urllib.parse import urlparse, urlunparse

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def _normalize_hostname(host: str) -> str:
    """
    Normalize a hostname for comparison:
    - strip whitespace
    - lowercase
    - strip trailing dots
    - convert to IDNA (punycode) if needed
    """
    if host is None:
        return ""
    host = host.strip().lower().rstrip(".")
    try:
        return host.encode("idna").decode("ascii")
    except Exception:
        # If IDNA conversion fails, fall back to the lowered host
        return host


def _is_host_allowed(host: str) -> bool:
    """
    Check whether the given host is in the approved domain list,
    allowing exact matches or subdomains (e.g., a.b.partner.com).
    """
    if not host:
        return False
    normalized = _normalize_hostname(host)
    for allowed in VALID_REDIRECT_DOMAINS:
        allowed_norm = _normalize_hostname(allowed)
        if normalized == allowed_norm or normalized.endswith("." + allowed_norm):
            return True
    return False


def secure_user_redirect(url_input: str) -> str:
    """
    Accepts a URL provided by a user and securely processes the redirection.

    Args:
        url_input (str): The URL input provided by the user.

    Returns:
        str: A validated and safe-to-use URL for the redirection.

    Raises:
        ValueError: If the URL is invalid, uses an unsupported scheme,
                    or the domain of the provided URL is not in the list
                    of approved domains.
    """
    if not isinstance(url_input, str):
        raise ValueError("URL must be a string.")
    candidate = url_input.strip()
    if not candidate:
        raise ValueError("URL must not be empty.")

    parsed = urlparse(candidate)

    # Require explicit http/https to avoid schemes like javascript:, data:, etc.
    if parsed.scheme not in ("http", "https"):
        raise ValueError("URL must use http or https scheme.")

    # hostname excludes userinfo and brackets; safer for validation
    host = parsed.hostname
    if not _is_host_allowed(host):
        raise ValueError("Redirection to unapproved domain is not allowed.")

    # Rebuild a safe URL without userinfo to avoid ambiguity/spoofing
    normalized_host = _normalize_hostname(host)
    netloc = normalized_host
    if parsed.port:
        netloc = f"{normalized_host}:{parsed.port}"

    # Normalize path to avoid backslash-based ambiguities
    path = (parsed.path or "").replace("\\", "/")
    if path and not path.startswith("/"):
        path = "/" + path

    safe_url = urlunparse(
        (
            parsed.scheme,
            netloc,
            path,
            "",  # params (deprecated)
            parsed.query or "",
            parsed.fragment or "",
        )
    )
    return safe_url
