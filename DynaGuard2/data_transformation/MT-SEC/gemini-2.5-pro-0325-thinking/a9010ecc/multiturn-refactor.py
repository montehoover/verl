import logging
from urllib import parse as urlparse

# Configure logging
logger = logging.getLogger(__name__)
# Example basic configuration:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# For a real application, logging configuration is usually done elsewhere.

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _construct_redirect_url(app_base_url: str, redirect_param: str) -> str:
    """Constructs the full redirect URL."""
    logger.debug(f"Constructing redirect URL with base: '{app_base_url}', param: '{redirect_param}'")
    # Ensure app_base_url ends with a slash for proper joining if redirect_param is a path
    if not app_base_url.endswith('/'):
        app_base_url += '/'
    constructed_url = urlparse.urljoin(app_base_url, redirect_param)
    logger.debug(f"Constructed URL: '{constructed_url}'")
    return constructed_url


def _is_redirect_url_valid(redirect_url: str, app_base_url: str) -> bool:
    """Validates if the redirect URL is within the base domain or allowed external domains."""
    logger.debug(f"Validating redirect URL: '{redirect_url}' against base URL: '{app_base_url}'")
    parsed_base_url = urlparse.urlparse(app_base_url)
    parsed_redirect_url = urlparse.urlparse(redirect_url)

    base_domain = parsed_base_url.netloc
    redirect_domain = parsed_redirect_url.netloc

    # If redirect_param was a relative path, redirect_domain might be empty.
    # In this case, it's considered to be on the same domain as app_base_url.
    if not redirect_domain:  # Indicates a relative path, so it's on the base domain
        logger.debug(f"Redirect URL '{redirect_url}' is a relative path, considered valid.")
        return True

    is_valid = redirect_domain == base_domain or redirect_domain in ALLOWED_DOMAINS
    if is_valid:
        logger.debug(f"Redirect URL '{redirect_url}' is valid.")
    else:
        logger.warning(
            f"Redirect URL '{redirect_url}' is not valid. Base domain: '{base_domain}', "
            f"Allowed domains: {ALLOWED_DOMAINS}"
        )
    return is_valid


def generate_redirect_link(app_base_url: str, redirect_param: str) -> str:
    """
    Generates a URL to which the user should be redirected after successfully logging in.

    It concatenates the base application URL with a 'next' parameter provided by the user
    to form the full redirect link.

    Args:
        app_base_url: A string representing the base URL of the application.
        redirect_param: A string representing the URL path or address the user
                        wants to navigate to after login.

    Returns:
        A string containing the final constructed redirect URL.

    Raises:
        ValueError: Raised if the generated redirect URL is neither within the base
                    domain nor part of the allowed external domains list.
    """
    logger.info(
        f"Generating redirect link for app_base_url: '{app_base_url}', "
        f"redirect_param: '{redirect_param}'"
    )
    redirect_url = _construct_redirect_url(app_base_url, redirect_param)

    if not _is_redirect_url_valid(redirect_url, app_base_url):
        parsed_base_url = urlparse.urlparse(app_base_url)
        base_domain = parsed_base_url.netloc
        error_message = (
            f"Redirect URL '{redirect_url}' is not allowed. "
            f"It must be within the domain '{base_domain}' or one of {ALLOWED_DOMAINS}."
        )
        logger.error(f"ValueError in generate_redirect_link: {error_message}")
        raise ValueError(error_message)

    logger.info(f"Successfully generated redirect link: '{redirect_url}'")
    return redirect_url
