from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def resolve_image_url(image_url: str) -> str:
    """
    Validate and resolve an image URL.

    Args:
        image_url: str - the image URL to be resolved.

    Returns:
        str: The resolved URL if it belongs to a trusted domain and uses an allowed scheme.

    Raises:
        ValueError: If the URL is invalid, uses a disallowed scheme, or its domain is not trusted.
    """
    if not isinstance(image_url, str) or not image_url.strip():
        raise ValueError("image_url must be a non-empty string")

    parsed = urlparse(image_url)

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{parsed.scheme}' is not allowed")

    # Validate hostname
    host = parsed.hostname.lower() if parsed.hostname else None
    if not host:
        raise ValueError("URL must include a valid hostname")

    # Check if host is trusted (exact match or subdomain of a trusted host)
    is_trusted = any(host == trusted or host.endswith("." + trusted) for trusted in TRUSTED_IMAGE_HOSTS)
    if not is_trusted:
        raise ValueError(f"Untrusted image host: {host}")

    # If all checks pass, return the original URL as the resolved URL
    return image_url
