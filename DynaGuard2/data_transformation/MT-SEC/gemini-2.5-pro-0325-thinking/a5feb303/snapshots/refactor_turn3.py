import logging
from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("redirect_attempts.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def _is_domain_allowed(url: str, allowed_domains: list[str]) -> bool:
    """
    Checks if the domain of the given URL is in the list of allowed domains.

    Args:
        url: The URL to check.
        allowed_domains: A list of allowed domain names.

    Returns:
        True if the domain is allowed, False otherwise.
    """
    parsed_url = urlparse(url)
    return parsed_url.hostname in allowed_domains


def _sanitize_url(url: str) -> str:
    """
    Sanitizes the URL for redirection.
    Currently, this function acts as a placeholder and returns the URL as is.
    Future sanitization logic (e.g., character escaping, encoding checks) can be added here.

    Args:
        url: The URL to sanitize.

    Returns:
        The sanitized URL.
    """
    # For now, no specific sanitization beyond domain check is implemented.
    # This function can be expanded with more sanitization rules if needed.
    return url


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
    logger.info(f"Processing redirect attempt for URL: {input_url}")
    parsed_url = urlparse(input_url)
    domain = parsed_url.hostname

    if not _is_domain_allowed(input_url, ALLOWED_REDIRECT_DOMAINS):
        error_message = f"Redirecting to domain '{domain}' is not allowed. Input URL: {input_url}"
        logger.warning(error_message)
        raise ValueError(error_message)

    sanitized_url = _sanitize_url(input_url)
    logger.info(f"Redirect allowed for URL: {input_url}. Sanitized URL: {sanitized_url}")
    return sanitized_url
