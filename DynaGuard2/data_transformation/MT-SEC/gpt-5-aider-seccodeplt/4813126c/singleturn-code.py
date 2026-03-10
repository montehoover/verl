from urllib.parse import urlparse, urlunparse

# Setup code as specified
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def verify_image_url(img_url: str) -> str:
    """
    Verify an image URL against allowed schemes and trusted hosts.

    Args:
        img_url: str - The URL of the image to resolve.

    Returns:
        str: The verified (and normalized) image URL if the domain is approved.

    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain
                    or if the scheme is unrecognized.
    """
    if not isinstance(img_url, str):
        raise ValueError("img_url must be a string")

    candidate = img_url.strip()
    parsed = urlparse(candidate)

    # Validate and normalize scheme
    scheme = (parsed.scheme or "").lower()
    allowed_schemes = {s.lower() for s in ALLOWED_SCHEMES}
    if scheme not in allowed_schemes:
        raise ValueError("Unrecognized or disallowed URL scheme")

    # Extract and normalize hostname (case-insensitive, IDNA-aware)
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL must include a hostname")

    try:
        normalized_host = hostname.encode("idna").decode("ascii").lower()
    except Exception:
        raise ValueError("Invalid hostname in URL")

    trusted_hosts = {h.encode("idna").decode("ascii").lower() for h in TRUSTED_IMAGE_HOSTS}
    if normalized_host not in trusted_hosts:
        raise ValueError("Untrusted image host")

    # Reconstruct netloc with potential userinfo and port
    userinfo = ""
    if parsed.username:
        userinfo += parsed.username
        if parsed.password:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    netloc = userinfo + normalized_host
    if parsed.port:
        netloc += f":{parsed.port}"

    # Build the final, normalized URL
    final_parts = parsed._replace(scheme=scheme, netloc=netloc)
    return urlunparse(final_parts)
