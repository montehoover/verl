import logging
from urllib.parse import urlparse

# Configure basic logging
# In a real application, this would likely be configured globally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    logging.info(f"Attempting to validate URL for redirection: '{url_input}'")

    if not url_input:
        logging.error(f"Validation failed for URL '{url_input}': URL input cannot be empty.")
        raise ValueError("URL input cannot be empty.")

    try:
        parsed_url = urlparse(url_input)
        domain = parsed_url.netloc
    except Exception as e:
        # Catch any parsing errors, though urlparse is generally robust
        logging.error(f"Validation failed for URL '{url_input}': Invalid URL format. Error: {e}")
        raise ValueError(f"Invalid URL format: {url_input}") from e

    if not domain: # Handle cases like "path/only" or if urlparse fails to find a netloc
        logging.error(f"Validation failed for URL '{url_input}': Could not extract domain.")
        raise ValueError(f"Could not extract domain from URL: {url_input}")

    if not _is_valid_redirect_domain(domain, VALID_REDIRECT_DOMAINS):
        error_message = (
            f"Redirect domain '{domain}' is not approved. "
            f"Approved domains are: {', '.join(VALID_REDIRECT_DOMAINS)}."
        )
        logging.error(f"Validation failed for URL '{url_input}': {error_message}")
        raise ValueError(error_message)

    logging.info(f"URL '{url_input}' validated successfully. Redirecting to '{url_input}'.")
    return url_input
