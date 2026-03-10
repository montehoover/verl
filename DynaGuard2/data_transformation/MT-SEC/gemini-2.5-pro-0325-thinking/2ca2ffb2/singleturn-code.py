from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def fetch_image_url(url: str) -> str:
    """
    Receives an image's URL as input and returns the final URL after ensuring
    the domain belongs to a list of trusted image hosting services.

    Args:
        url: str, the URL of the image to resolve.

    Returns:
        The verified image URL if the domain is approved.

    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain
                    or if the scheme is unrecognized.
    """
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}. Allowed schemes are: {ALLOWED_SCHEMES}")

    if parsed_url.hostname not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Domain not trusted: {parsed_url.hostname}. Trusted domains are: {TRUSTED_IMAGE_HOSTS}")

    return url
