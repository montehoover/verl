from urllib.parse import urlparse

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def secure_user_redirect(url_input: str) -> str:
    """
    Accepts a URL provided by a user and securely processes the redirection.

    Args:
        url_input: The URL input provided by the user.

    Returns:
        A validated and safe-to-use URL for the redirection.

    Raises:
        ValueError: If the domain of the provided URL is not in the list of
                    approved domains.
    """
    parsed_url = urlparse(url_input)
    domain = parsed_url.netloc

    if domain in VALID_REDIRECT_DOMAINS:
        return url_input
    else:
        raise ValueError(
            f"Invalid redirect domain: {domain}. "
            f"Must be one of {VALID_REDIRECT_DOMAINS}"
        )
