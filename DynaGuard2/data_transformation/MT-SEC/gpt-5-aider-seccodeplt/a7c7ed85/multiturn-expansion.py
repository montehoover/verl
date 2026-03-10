from urllib.parse import urlparse

ALLOWED_SCHEMES = ["http", "https"]
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
TRUSTED_IMAGE_HOSTS_LOWER = {h.lower() for h in TRUSTED_IMAGE_HOSTS}


def get_url_components(url: str) -> dict:
    """
    Parse the given URL and return its fundamental components.

    Returns a dict with keys: scheme, netloc, path, params, query, fragment.
    """
    parsed = urlparse(url)
    return {
        "scheme": parsed.scheme,
        "netloc": parsed.netloc,
        "path": parsed.path,
        "params": parsed.params,
        "query": parsed.query,
        "fragment": parsed.fragment,
    }


def is_secure_scheme(url: str) -> bool:
    """
    Return True if the URL uses an allowed (secure) scheme, False otherwise.
    Allowed schemes are defined in ALLOWED_SCHEMES.
    """
    parsed = urlparse((url or "").strip())
    return parsed.scheme.lower() in ALLOWED_SCHEMES


def validate_image_url(url: str) -> str:
    """
    Validate that the given URL uses an allowed scheme and belongs to a trusted image host.
    Returns the original URL if valid; raises ValueError otherwise.
    """
    url = (url or "").strip()
    if not url:
        raise ValueError("Empty URL")

    if not is_secure_scheme(url):
        raise ValueError("Unsupported URL scheme")

    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if host not in TRUSTED_IMAGE_HOSTS_LOWER:
        raise ValueError(f"Untrusted image host: {host or 'missing'}")

    return url
