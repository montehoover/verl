from urllib.parse import urlparse, urlunparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def _is_allowed_domain(hostname: str) -> bool:
    """
    Returns True if the given hostname is exactly an allowed domain or a subdomain of one.
    """
    hostname = hostname.lower().strip(".")
    for allowed in ALLOWED_REDIRECT_DOMAINS:
        allowed = allowed.lower().strip(".")
        if hostname == allowed:
            return True
        if hostname.endswith("." + allowed):
            return True
    return False


def process_user_redirect(input_url: str) -> str:
    """
    Accepts a user-provided URL and returns a sanitized URL safe for redirection.

    Args:
        input_url (str): The URL provided by the user for redirection.

    Returns:
        str: A sanitized URL that is considered safe for redirection.

    Raises:
        ValueError: If the URL is invalid, uses an unsupported scheme, or the domain
                    is not part of the allowed domains.
    """
    if not isinstance(input_url, str) or not input_url.strip():
        raise ValueError("A non-empty URL string must be provided.")

    raw = input_url.strip()

    parsed = urlparse(raw)

    # Require absolute URLs with http/https scheme
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("Only http and https schemes are allowed.")

    # Ensure we have a hostname
    hostname = parsed.hostname  # lowercased by urlparse
    if not hostname:
        raise ValueError("URL must contain a valid hostname.")

    # Validate allowed domains (exact or subdomain)
    if not _is_allowed_domain(hostname):
        raise ValueError("Redirection to the specified domain is not allowed.")

    # Rebuild a sanitized URL:
    # - Drop userinfo if present
    # - Preserve port if explicitly provided
    netloc = hostname
    if parsed.port:
        # urlparse.port validates numeric range; if invalid it would raise on access.
        netloc = f"{hostname}:{parsed.port}"

    sanitized = urlunparse((
        parsed.scheme.lower(),
        netloc,
        parsed.path or "",
        parsed.params or "",
        parsed.query or "",
        parsed.fragment or "",
    ))

    return sanitized
