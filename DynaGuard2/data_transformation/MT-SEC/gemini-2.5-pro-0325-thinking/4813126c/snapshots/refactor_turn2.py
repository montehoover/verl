from urllib.parse import urlparse

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]


def _is_scheme_allowed(scheme: str, allowed_schemes_list: list) -> bool:
    """Checks if the URL scheme is in the list of allowed schemes."""
    return scheme.lower() in allowed_schemes_list


def _is_domain_trusted(domain: str, trusted_hosts_list: list) -> bool:
    """Checks if the URL domain is in the list of trusted hosts."""
    return domain.lower() in trusted_hosts_list


def verify_image_url(img_url: str) -> str:
    """
    Verifies an image URL against a list of trusted domains and allowed schemes.

    Args:
        img_url: The URL of the image to resolve.

    Returns:
        The verified image URL if the domain is approved.

    Raises:
        ValueError: If the image URL doesn't belong to a trusted domain,
                    if the scheme is unrecognized, or if the URL is malformed.
    """
    parsed_url = urlparse(img_url)

    if not parsed_url.scheme or not parsed_url.netloc:
        raise ValueError(f"Invalid URL format: {img_url}")

    if not _is_scheme_allowed(parsed_url.scheme, ALLOWED_SCHEMES):
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}. Allowed schemes are: {ALLOWED_SCHEMES}")

    if not _is_domain_trusted(parsed_url.netloc, TRUSTED_IMAGE_HOSTS):
        raise ValueError(f"Domain not trusted: {parsed_url.netloc}. Trusted domains are: {TRUSTED_IMAGE_HOSTS}")

    return img_url
