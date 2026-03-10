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

def secure_user_redirect(url_string):
    """
    Ensures a URL is from an approved domain before processing redirection.

    Args:
        url_string: The URL to validate.

    Returns:
        The validated URL if it's safe.

    Raises:
        ValueError: If the domain is not approved.
    """
    if not is_valid_domain(url_string):
        raise ValueError(f"Redirect URL domain not approved: {url_string}")
    return url_string
