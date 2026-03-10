from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]
_TRUSTED_HOSTS_LOWER = {h.lower() for h in TRUSTED_IMAGE_HOSTS}

def verify_image_url(img_url: str) -> str:
    """
    Validate that the URL uses an allowed scheme (http/https) and the hostname is in TRUSTED_IMAGE_HOSTS.
    Returns the original URL string if valid; raises ValueError otherwise.
    """
    if not isinstance(img_url, str):
        raise ValueError("img_url must be a string.")

    s = img_url.strip()
    if not s:
        raise ValueError("img_url cannot be empty.")

    parsed = urlparse(s)

    scheme_lower = parsed.scheme.lower()
    if scheme_lower not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or '(none)'}")

    host = parsed.hostname
    if not host:
        raise ValueError("URL must include a host.")

    if host.lower() not in _TRUSTED_HOSTS_LOWER:
        raise ValueError(f"Untrusted image host: {host}")

    return s
