from urllib.parse import urlparse
ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def user_defined_redirect(redirect_url: str) -> str:
    """
    Accepts a URL input from the user and redirects to it in a secure manner.

    Args:
        redirect_url: str, the URL provided by the user for redirection.

    Returns:
        A sanitized URL that is considered safe for redirection.

    Raises:
        ValueError: If the domain of the provided URL is not part of the allowed domains.
    """
    parsed_url = urlparse(redirect_url)
    domain = parsed_url.netloc

    # Ensure the scheme is http or https to prevent other schemes like javascript:
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only http and https are allowed.")

    if domain in ALLOWED_REDIRECT_DOMAINS:
        return redirect_url
    else:
        # Check for subdomains of allowed domains if that's a desired behavior.
        # For this implementation, we'll stick to exact domain matches.
        # Example: if "example.com" is allowed, "sub.example.com" would not be by default.
        # If subdomains should be allowed, the check would need to be more sophisticated,
        # e.g., domain.endswith(tuple(f".{d}" for d in ALLOWED_REDIRECT_DOMAINS) + tuple(ALLOWED_REDIRECT_DOMAINS))
        raise ValueError(f"Redirect to domain '{domain}' is not allowed.")
