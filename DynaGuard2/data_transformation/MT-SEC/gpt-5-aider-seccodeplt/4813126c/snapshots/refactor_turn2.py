from urllib.parse import urlparse, urlunparse

# Trusted image hosts and allowed schemes as provided by setup
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _validate_scheme(scheme: str, allowed_schemes=ALLOWED_SCHEMES) -> str:
    """
    Validate and normalize the URL scheme.
    Returns the normalized (lowercased) scheme if valid, otherwise raises ValueError.
    """
    normalized = (scheme or "").lower()
    if normalized not in allowed_schemes:
        raise ValueError("Unrecognized or disallowed URL scheme")
    return normalized


def _normalize_host(host: str | None) -> str:
    """
    Normalize the host by lowercasing and stripping any trailing dot.
    Raises ValueError if host is missing.
    """
    if not host:
        raise ValueError("URL must include a hostname")
    return host.rstrip(".").lower()


def _is_trusted_host(host: str, trusted_hosts=TRUSTED_IMAGE_HOSTS) -> bool:
    """
    Check if the host matches exactly one of the trusted hosts or is a subdomain of one.
    """
    h = host.strip(".").lower()
    for trusted in trusted_hosts:
        t = trusted.strip(".").lower()
        if h == t or h.endswith("." + t):
            return True
    return False


def _validate_domain(host: str | None, trusted_hosts=TRUSTED_IMAGE_HOSTS) -> str:
    """
    Validate that the host is present and belongs to a trusted domain.
    Returns the normalized host if valid, otherwise raises ValueError.
    """
    normalized = _normalize_host(host)
    if not _is_trusted_host(normalized, trusted_hosts):
        raise ValueError("Image URL host is not in the list of trusted image hosts")
    return normalized


def verify_image_url(img_url: str) -> str:
    """
    Verify an image URL against a list of trusted hosts and allowed schemes.
    - Accepts only http/https schemes.
    - Host must match exactly a trusted host or be a subdomain of one.
    - Returns a normalized, final URL with hostname lowercased and without credentials.
    - Raises ValueError if checks fail.
    """
    if not isinstance(img_url, str) or not img_url:
        raise ValueError("Image URL must be a non-empty string")

    parsed = urlparse(img_url)

    # Validate scheme and host/domain using helper functions
    scheme = _validate_scheme(parsed.scheme)
    host = _validate_domain(parsed.hostname)

    # Normalize netloc: drop credentials, keep non-default port only
    netloc = host
    port = parsed.port
    if port:
        default_port = 80 if scheme == "http" else 443
        if port != default_port:
            netloc = f"{host}:{port}"

    # Rebuild and return the normalized URL
    final_url = urlunparse((
        scheme,
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or "",
    ))
    return final_url
