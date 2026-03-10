from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]  # List of trusted domains for redirection


def user_defined_redirect(redirect_url: str) -> str:
    """
    Validates a user-provided URL to ensure it is safe for redirection.

    The function parses the input URL to extract its domain (hostname).
    It then checks if this domain is present in the `ALLOWED_REDIRECT_DOMAINS` list.

    Args:
        redirect_url: The URL string to be validated.
                      Example: "http://trusted-redirect.com/some/path"

    Returns:
        The original `redirect_url` string if its domain is found in the
        `ALLOWED_REDIRECT_DOMAINS` list. The function does not modify the URL.

    Raises:
        ValueError: If the URL's domain is not in the `ALLOWED_REDIRECT_DOMAINS` list,
                    or if the domain cannot be extracted from the `redirect_url`
                    (e.g., due to a malformed URL resulting in `parsed_url.hostname`
                    being None or empty).
    """
    # Parse the URL to safely extract its components.
    # `urlparse` is generally robust and handles various URL formats.
    parsed_url = urlparse(redirect_url)
    domain = parsed_url.hostname  # Extracts the domain, e.g., "trusted-redirect.com"

    # Validate the extracted domain.
    # The domain must exist (not None, not an empty string) and be present
    # in the list of allowed domains.
    if domain and domain in ALLOWED_REDIRECT_DOMAINS:
        return redirect_url
    else:
        # If the domain is not allowed or could not be parsed, raise an error.
        # Construct a clear and informative error message.
        allowed_domains_str = ", ".join(ALLOWED_REDIRECT_DOMAINS)
        if domain:
            # Case: Domain was extracted but is not in the allowed list.
            error_message = (
                f"Redirect to domain '{domain}' is not allowed. "
                f"Allowed domains are: {allowed_domains_str}."
            )
        else:
            # Case: Domain could not be extracted from the URL (e.g., malformed URL).
            error_message = (
                f"Could not determine a valid domain from the URL '{redirect_url}'. "
                f"Redirection is only allowed to the following domains: {allowed_domains_str}."
            )
        raise ValueError(error_message)
