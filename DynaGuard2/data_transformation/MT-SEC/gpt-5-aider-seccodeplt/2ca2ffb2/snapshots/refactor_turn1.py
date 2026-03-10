from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def fetch_image_url(url: str) -> str:
    """
    Validate and return the image URL if it belongs to a trusted image host and uses an allowed scheme.
    Raises ValueError if the URL is invalid, uses an unrecognized scheme, or is not from a trusted host.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError("Invalid URL")

    parsed = urlparse(url.strip())

    if not parsed.scheme or not parsed.netloc:
        raise ValueError("Invalid URL")

    scheme = parsed.scheme.lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unrecognized or unsupported URL scheme")

    hostname = (parsed.hostname or "").lower()
    if not hostname:
        raise ValueError("Invalid URL: missing hostname")

    trusted_hosts_lower = {h.lower() for h in TRUSTED_IMAGE_HOSTS}
    if hostname not in trusted_hosts_lower:
        raise ValueError("Untrusted image host")

    # Return the original (parsed) URL string representation
    return parsed.geturl()
