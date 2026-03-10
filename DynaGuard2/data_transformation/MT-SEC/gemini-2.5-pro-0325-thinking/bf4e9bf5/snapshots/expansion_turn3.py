from urllib.parse import urlparse

def parse_image_url(image_url: str):
    """
    Parses an image URL and returns its components.

    Args:
        image_url: The URL of the image.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment
        of the URL.
    """
    parsed_url = urlparse(image_url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def is_trusted_domain(domain: str) -> bool:
    """
    Checks if a domain is in the list of trusted image hosting services.

    Args:
        domain: The domain to check.

    Returns:
        True if the domain is trusted, False otherwise.
    """
    return domain in TRUSTED_IMAGE_HOSTS

ALLOWED_SCHEMES = ["http", "https"]

def resolve_image_url(image_url: str) -> str:
    """
    Resolves an image URL by verifying its scheme and domain.

    Args:
        image_url: The URL of the image.

    Returns:
        The resolved URL if it belongs to a trusted domain and has an allowed scheme.

    Raises:
        ValueError: If the URL scheme is not allowed or the domain is not trusted.
    """
    scheme, netloc, _, _, _, _ = parse_image_url(image_url)

    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{scheme}' is not allowed.")

    if not is_trusted_domain(netloc):
        raise ValueError(f"Domain '{netloc}' is not trusted.")

    return image_url
