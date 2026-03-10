from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_url_redirect(input_url: str) -> str:
    """
    Accepts a URL provided by a user and securely processes the redirection.

    Args:
        input_url: The URL input provided by the user.

    Returns:
        A validated and safe-to-use URL for the redirection.

    Raises:
        ValueError: If the domain of the provided URL is not in the list of
                    approved domains.
    """
    parsed_url = urlparse(input_url)
    domain = parsed_url.netloc
    if domain in AUTHORIZED_REDIRECT_DOMAINS:
        return input_url
    else:
        raise ValueError(
            f"Redirect to domain '{domain}' is not authorized. "
            f"Authorized domains are: {', '.join(AUTHORIZED_REDIRECT_DOMAINS)}"
        )
