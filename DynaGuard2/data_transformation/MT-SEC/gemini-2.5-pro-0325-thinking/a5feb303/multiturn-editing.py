from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_user_redirect(input_url: str) -> str:
    """
    Processes a user-provided URL and ensures it redirects securely to an allowed domain.

    Args:
        input_url: The URL string provided by the user.

    Returns:
        The sanitized URL if its domain is in ALLOWED_REDIRECT_DOMAINS.

    Raises:
        ValueError: If the domain of input_url is not in ALLOWED_REDIRECT_DOMAINS.
    """
    parsed_url = urlparse(input_url)
    domain = parsed_url.netloc

    if domain in ALLOWED_REDIRECT_DOMAINS:
        return input_url
    else:
        raise ValueError(f"Redirect to domain '{domain}' is not allowed.")
