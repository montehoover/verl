from urllib.parse import urlparse

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def _is_valid_redirect_domain(domain: str, approved_domains: list[str]) -> bool:
    """
    Checks if the given domain is in the list of approved redirect domains.

    Args:
        domain: The domain string to validate.
        approved_domains: A list of approved domain strings.

    Returns:
        True if the domain is approved, False otherwise.
    """
    return domain in approved_domains

def secure_user_redirect(url_input: str) -> str:
    """
    Validates a user-provided URL for redirection against a list of approved domains.

    Args:
        url_input: The URL string provided by the user.

    Returns:
        The validated URL string if its domain is approved.

    Raises:
        ValueError: If the domain of the provided URL is not in the
                    VALID_REDIRECT_DOMAINS list or if the URL is malformed.
    """
    if not url_input:
        raise ValueError("URL input cannot be empty.")

    try:
        parsed_url = urlparse(url_input)
        domain = parsed_url.netloc
    except Exception as e:
        # Catch any parsing errors, though urlparse is generally robust
        raise ValueError(f"Invalid URL format: {url_input}") from e

    if not domain: # Handle cases like "path/only" or if urlparse fails to find a netloc
        raise ValueError(f"Could not extract domain from URL: {url_input}")

    if not _is_valid_redirect_domain(domain, VALID_REDIRECT_DOMAINS):
        raise ValueError(
            f"Redirect domain '{domain}' is not approved. "
            f"Approved domains are: {', '.join(VALID_REDIRECT_DOMAINS)}."
        )

    return url_input
