from urllib.parse import urlparse
from typing import List

def check_trusted_domain(url_string: str, trusted_domains: List[str]) -> bool:
    """
    Extracts the domain from a URL string and checks if it's in a list of trusted domains.

    Args:
        url_string: The URL string to parse.
        trusted_domains: A list of trusted domain strings.

    Returns:
        True if the URL's domain is in the trusted list (safe), False otherwise (unsafe).
    """
    parsed_url = urlparse(url_string)
    domain = parsed_url.netloc
    return domain in trusted_domains
