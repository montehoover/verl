from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def _is_domain_authorized(domain: str, authorized_domains: list[str]) -> bool:
    """
    Checks if a domain is in the list of authorized domains.

    Args:
        domain: The domain string to check.
        authorized_domains: A list of authorized domain strings.

    Returns:
        True if the domain is authorized, False otherwise.
    """
    return domain in authorized_domains


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
    if not _is_domain_authorized(domain, AUTHORIZED_REDIRECT_DOMAINS):
        raise ValueError(
            f"Redirecting to domain '{domain}' is not allowed. "
            f"Must be one of: {', '.join(AUTHORIZED_REDIRECT_DOMAINS)}"
        )
    return input_url
