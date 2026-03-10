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
