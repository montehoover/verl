from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_url_redirect(input_url: str) -> str:
    """
    Validates a user-provided URL for secure redirection.

    Args:
        input_url: The URL string provided by the user.

    Returns:
        The validated and safe-to-use URL for redirection.

    Raises:
        ValueError: If the domain of the provided URL is not in
                    the list of approved domains.
    """
    parsed_url = urlparse(input_url)
    domain = parsed_url.netloc
    if domain not in AUTHORIZED_REDIRECT_DOMAINS:
        raise ValueError(
            f"Redirecting to domain '{domain}' is not allowed. "
            f"Must be one of: {', '.join(AUTHORIZED_REDIRECT_DOMAINS)}"
        )
    return input_url
