from urllib.parse import urlparse, urlunparse

# Trusted image hosts and allowed schemes as provided by setup
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


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

    # Validate scheme
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unrecognized or disallowed URL scheme")

    # Extract and normalize host
    host = parsed.hostname  # excludes credentials and brackets around IPv6
    if not host:
        raise ValueError("URL must include a hostname")
    host = host.rstrip(".").lower()

    # Determine trust by exact match or subdomain match
    def is_trusted(h: str) -> bool:
        for trusted in TRUSTED_IMAGE_HOSTS:
            t = trusted.strip(".").lower()
            if h == t or h.endswith("." + t):
                return True
        return False

    if not is_trusted(host):
        raise ValueError("Image URL host is not in the list of trusted image hosts")

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
