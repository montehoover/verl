from urllib.parse import urlparse

TRUSTED_DOMAINS = [
    "example.com",
    "google.com",
    "openai.com",
]

def is_trusted_domain(url_string):
    """
    Checks if the domain of a URL string is in a list of trusted domains.

    Args:
        url_string: The URL string.

    Returns:
        True if the domain is trusted, False otherwise.
    """
    parsed_url = urlparse(url_string)
    domain = parsed_url.netloc
    return domain in TRUSTED_DOMAINS
