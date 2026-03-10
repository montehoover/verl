from urllib.parse import urlparse

def parse_image_url(url_string):
    """
    Parses an image URL and returns its components.

    Args:
        url_string: The URL to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url_string)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def is_trusted_domain(url_string):
    """
    Checks if the domain of a URL is in a list of trusted domains.

    Args:
        url_string: The URL to check.

    Returns:
        True if the domain is trusted, False otherwise.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.netloc in TRUSTED_IMAGE_HOSTS

ALLOWED_SCHEMES = ["http", "https"]

def fetch_image_url(url_string):
    """
    Verifies an image URL against trusted domains and allowed schemes.

    Args:
        url_string: The URL to verify.

    Returns:
        The verified URL string if valid.

    Raises:
        ValueError: If the domain is not trusted or the scheme is unrecognized.
    """
    parsed_url = urlparse(url_string)

    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}. Allowed schemes are: {ALLOWED_SCHEMES}")

    if parsed_url.netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"Domain not trusted: {parsed_url.netloc}. Trusted domains are: {TRUSTED_IMAGE_HOSTS}")

    return url_string
