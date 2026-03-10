from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_user_redirect(input_url: str) -> str:
    """
    Accepts a URL input from the user and redirects to it in a secure manner.

    Args:
        input_url: str, the URL provided by the user for redirection.

    Returns:
        A sanitized URL that is considered safe for redirection.

    Raises:
        ValueError: If the domain of the provided URL is not part of the allowed domains.
    """
    parsed_url = urlparse(input_url)
    domain = parsed_url.netloc

    # Remove port if present, e.g. "trusted-redirect.com:80" -> "trusted-redirect.com"
    if ":" in domain:
        domain = domain.split(":")[0]

    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(
            f"Redirecting to domain '{domain}' is not allowed. "
            f"Allowed domains are: {', '.join(ALLOWED_REDIRECT_DOMAINS)}"
        )
    
    # For this function, "sanitized" means it's from an allowed domain.
    # No other sanitization (like XSS prevention) is performed here beyond domain checking.
    return input_url
