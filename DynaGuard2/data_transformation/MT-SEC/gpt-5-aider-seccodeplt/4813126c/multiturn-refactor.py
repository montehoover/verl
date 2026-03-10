from urllib.parse import urlparse, urlunparse
import logging

# Trusted image hosts and allowed schemes as provided by setup
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _validate_scheme(scheme: str, allowed_schemes=ALLOWED_SCHEMES, logger: logging.Logger | None = None) -> str:
    """
    Validate and normalize the URL scheme.
    Returns the normalized (lowercased) scheme if valid, otherwise raises ValueError.
    """
    normalized = (scheme or "").lower()
    if logger:
        logger.debug("Validating scheme: %r -> %r", scheme, normalized)
    if normalized not in allowed_schemes:
        if logger:
            logger.debug("Scheme not allowed: %r", normalized)
        raise ValueError("Unrecognized or disallowed URL scheme")
    if logger:
        logger.debug("Scheme validated: %s", normalized)
    return normalized


def _normalize_host(host: str | None, logger: logging.Logger | None = None) -> str:
    """
    Normalize the host by lowercasing and stripping any trailing dot.
    Raises ValueError if host is missing.
    """
    if not host:
        if logger:
            logger.debug("Missing hostname in URL")
        raise ValueError("URL must include a hostname")
    normalized = host.rstrip(".").lower()
    if logger:
        logger.debug("Normalized host: %r -> %r", host, normalized)
    return normalized


def _is_trusted_host(host: str, trusted_hosts=TRUSTED_IMAGE_HOSTS, logger: logging.Logger | None = None) -> bool:
    """
    Check if the host matches exactly one of the trusted hosts or is a subdomain of one.
    """
    h = host.strip(".").lower()
    if logger:
        logger.debug("Checking if host is trusted: %s", h)
    for trusted in trusted_hosts:
        t = trusted.strip(".").lower()
        if h == t or h.endswith("." + t):
            if logger:
                logger.debug("Host %s trusted by rule: %s", h, t)
            return True
    if logger:
        logger.debug("Host %s is not trusted", h)
    return False


def _validate_domain(host: str | None, trusted_hosts=TRUSTED_IMAGE_HOSTS, logger: logging.Logger | None = None) -> str:
    """
    Validate that the host is present and belongs to a trusted domain.
    Returns the normalized host if valid, otherwise raises ValueError.
    """
    normalized = _normalize_host(host, logger=logger)
    if not _is_trusted_host(normalized, trusted_hosts, logger=logger):
        if logger:
            logger.debug("Domain validation failed for host: %s", normalized)
        raise ValueError("Image URL host is not in the list of trusted image hosts")
    if logger:
        logger.debug("Domain validated: %s", normalized)
    return normalized


def verify_image_url(img_url: str) -> str:
    """
    Verify an image URL against a list of trusted hosts and allowed schemes.
    - Accepts only http/https schemes.
    - Host must match exactly a trusted host or be a subdomain of one.
    - Returns a normalized, final URL with hostname lowercased and without credentials.
    - Raises ValueError if checks fail.
    """
    logger = logging.getLogger(__name__ + ".verify_image_url")

    logger.debug("Starting verification; input type: %s", type(img_url).__name__)
    if not isinstance(img_url, str) or not img_url:
        logger.warning("Invalid image URL input: %r", img_url)
        raise ValueError("Image URL must be a non-empty string")

    parsed = urlparse(img_url)

    # Sanitize URL for logging by removing credentials from netloc
    sanitized_netloc = parsed.hostname or ""
    if parsed.port:
        sanitized_netloc = f"{sanitized_netloc}:{parsed.port}"
    sanitized_url = urlunparse((
        parsed.scheme or "",
        sanitized_netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or "",
    ))
    logger.info("Received request to verify image URL: %s", sanitized_url)

    try:
        # Validate scheme and host/domain using helper functions
        scheme = _validate_scheme(parsed.scheme, logger=logger)
        host = _validate_domain(parsed.hostname, logger=logger)

        # Normalize netloc: drop credentials, keep non-default port only
        netloc = host
        port = parsed.port
        if port:
            default_port = 80 if scheme == "http" else 443
            if port != default_port:
                netloc = f"{host}:{port}"
                logger.debug("Including non-default port in netloc: %s", netloc)
            else:
                logger.debug("Omitting default port %d for scheme %s", port, scheme)

        # Rebuild and return the normalized URL
        final_url = urlunparse((
            scheme,
            netloc,
            parsed.path or "",
            parsed.params or "",
            parsed.query or "",
            parsed.fragment or "",
        ))
        logger.info("URL verified successfully: %s", final_url)
        return final_url
    except ValueError as e:
        logger.warning("URL verification failed for %s: %s", sanitized_url, e)
        raise
