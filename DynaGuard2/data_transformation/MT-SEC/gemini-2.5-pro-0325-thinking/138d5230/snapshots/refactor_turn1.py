from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def user_defined_redirect(redirect_url: str) -> str:
    """
    Accepts a URL input from the user and redirects to it in a secure manner.

    Args:
        redirect_url: The URL provided by the user for redirection.

    Returns:
        The sanitized URL that is considered safe for redirection.

    Raises:
        ValueError: If the domain of the provided URL is not part of the allowed domains.
    """
    parsed_url = urlparse(redirect_url)
    domain = parsed_url.hostname

    if domain and domain in ALLOWED_REDIRECT_DOMAINS:
        return redirect_url
    else:
        raise ValueError(
            f"Redirect to domain '{domain}' is not allowed. "
            f"Allowed domains are: {', '.join(ALLOWED_REDIRECT_DOMAINS)}"
        )
