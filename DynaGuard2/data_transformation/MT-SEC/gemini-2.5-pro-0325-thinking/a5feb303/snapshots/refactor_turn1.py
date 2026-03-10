from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_user_redirect(input_url: str) -> str:
    """
    Accepts a URL input from the user and redirects to it in a secure manner.

    Args:
        input_url: The URL provided by the user for redirection.

    Returns:
        The sanitized URL that is considered safe for redirection.

    Raises:
        ValueError: If the domain of the provided URL is not part of the allowed domains.
    """
    parsed_url = urlparse(input_url)
    if parsed_url.hostname not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(
            f"Redirecting to domain '{parsed_url.hostname}' is not allowed."
        )
    return input_url
