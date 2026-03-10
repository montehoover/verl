from urllib.parse import urlparse

def parse_image_url(url_string):
    """
    Parses a URL string and returns its components.

    Args:
        url_string: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, and query.
    """
    parsed_url = urlparse(url_string)
    return parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.query

ALLOWED_SCHEMES = ["http", "https"]

def check_url_safety(url_string):
    """
    Checks if a URL uses an unrecognized scheme.

    Args:
        url_string: The URL string to check.

    Returns:
        True if the URL has an unrecognized scheme, False otherwise.
    """
    scheme, _, _, _ = parse_image_url(url_string)
    if scheme not in ALLOWED_SCHEMES:
        return True
    return False

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]

def verify_image_url(image_url):
    """
    Verifies if the image URL is from a trusted domain and has an allowed scheme.

    Args:
        image_url: The URL string of the image.

    Returns:
        The verified image URL if it's safe and trusted.

    Raises:
        ValueError: If the URL scheme is not allowed or the host is not trusted.
    """
    scheme, netloc, _, _ = parse_image_url(image_url)

    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"URL scheme '{scheme}' is not allowed.")

    if netloc not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"URL host '{netloc}' is not a trusted image host.")

    return image_url
