from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validate that the URL uses an allowed scheme and belongs to a trusted image host.

    Parameters:
        img_url (str): The input image URL.

    Returns:
        str: The original URL if it passes validation.

    Raises:
        ValueError: If the scheme is unsupported/missing or the host is untrusted.
    """
    parsed = urlparse(img_url)

    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError("Unsupported or missing URL scheme. Allowed schemes are: http, https.")

    host = parsed.hostname.lower() if parsed.hostname else None
    if not host or host not in {h.lower() for h in TRUSTED_IMAGE_HOSTS}:
        raise ValueError("Untrusted image host.")

    return img_url
