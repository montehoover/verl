from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_url_components(url: str) -> tuple[str, str, str, str, str, str]:
    """
    Extracts components from a given URL.

    Args:
        url: The URL string to parse.

    Returns:
        A tuple containing the scheme, netloc, path, params, query, and fragment.
    """
    parsed_url = urlparse(url)
    return (
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        parsed_url.query,
        parsed_url.fragment,
    )

def is_domain_authorized(url: str) -> bool:
    """
    Checks if the domain of the given URL is in the list of authorized domains.

    Args:
        url: The URL string to check.

    Returns:
        True if the domain is authorized, False otherwise.
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc
    return domain in AUTHORIZED_REDIRECT_DOMAINS
