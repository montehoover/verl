from urllib.parse import urlparse

def extract_url_components(url_string):
    """
    Breaks down a URL into its components.

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

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def is_valid_domain(url_string):
    """
    Checks if the domain of a URL is in a predefined list of approved domains.

    Args:
        url_string: The URL to check.

    Returns:
        True if the domain is valid, False otherwise.
    """
    parsed_url = urlparse(url_string)
    domain = parsed_url.netloc
    return domain in VALID_REDIRECT_DOMAINS
