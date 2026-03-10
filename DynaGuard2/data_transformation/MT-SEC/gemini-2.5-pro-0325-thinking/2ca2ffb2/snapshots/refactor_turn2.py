from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _validate_scheme(scheme: str, allowed_schemes: list[str]):
    """
    Validates the URL scheme against a list of allowed schemes.

    Args:
        scheme: The scheme of the URL.
        allowed_schemes: A list of allowed schemes.

    Raises:
        ValueError: If the scheme is not in the list of allowed schemes.
    """
    if scheme not in allowed_schemes:
        raise ValueError(
            f"Unrecognized scheme: {scheme}. "
            f"Allowed schemes are: {', '.join(allowed_schemes)}"
        )


def _validate_domain(domain: str, trusted_hosts: list[str]):
    """
    Validates the URL domain against a list of trusted hosts.

    Args:
        domain: The domain (netloc) of the URL.
        trusted_hosts: A list of trusted host domains.

    Raises:
        ValueError: If the domain is not in the list of trusted hosts.
    """
    if domain not in trusted_hosts:
        raise ValueError(
            f"Untrusted domain: {domain}. "
            f"Trusted domains are: {', '.join(trusted_hosts)}"
        )


def fetch_image_url(url: str) -> str:
    """
    Resolves an image URL, ensuring the domain belongs to a list of trusted image hosting services.

    Args:
        url: The URL of the image to resolve.

    Returns:
        The verified image URL if the domain is approved.

    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain 
                    or if the scheme is unrecognized.
    """
    parsed_url = urlparse(url)

    _validate_scheme(parsed_url.scheme, ALLOWED_SCHEMES)
    _validate_domain(parsed_url.netloc, TRUSTED_IMAGE_HOSTS)

    return url
