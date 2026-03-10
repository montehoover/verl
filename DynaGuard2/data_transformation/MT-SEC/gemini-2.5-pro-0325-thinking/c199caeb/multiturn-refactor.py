import logging
from urllib import parse as urlparse

# Configure logging
logger = logging.getLogger(__name__)
# Example basic configuration:
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# For more advanced configuration, you might set this up outside this module.

APPROVED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _construct_final_url(app_url: str, redirect_param: str) -> tuple[str, urlparse.ParseResult]:
    """
    Constructs the final redirect URL from the application URL and redirect parameter.
    """
    logger.debug(f"Constructing final URL with app_url='{app_url}', redirect_param='{redirect_param}'")
    parsed_app_url = urlparse.urlparse(app_url)
    parsed_redirect_param = urlparse.urlparse(redirect_param)

    if parsed_redirect_param.scheme and parsed_redirect_param.netloc:
        logger.debug(f"Redirect param '{redirect_param}' is a full URL.")
        # redirect_param is a full URL
        final_url = redirect_param
        parsed_final_url = parsed_redirect_param
    else:
        logger.debug(f"Redirect param '{redirect_param}' is a path. Joining with app_url.")
        # redirect_param is a path, join it with app_url
        path_component = redirect_param
        if not path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            pass  # app_url ends with / and path_component doesn't start with /
        elif not path_component.startswith('/') and not parsed_app_url.path.endswith('/'):
            path_component = '/' + path_component
        elif path_component.startswith('/') and parsed_app_url.path.endswith('/'):
            path_component = path_component[1:]

        final_url = urlparse.urljoin(app_url, path_component)
        parsed_final_url = urlparse.urlparse(final_url)
    logger.info(f"Constructed final_url='{final_url}'")
    return final_url, parsed_final_url


def _validate_redirect_url(final_url: str, parsed_final_url: urlparse.ParseResult, app_domain: str, allowed_domains: list[str]) -> None:
    """
    Validates if the final redirect URL is within allowed domains.
    Raises ValueError if validation fails.
    """
    logger.debug(
        f"Validating final_url='{final_url}' (domain: {parsed_final_url.netloc}) "
        f"against app_domain='{app_domain}' and allowed_domains={allowed_domains}"
    )
    final_domain = parsed_final_url.netloc
    if not (final_domain == app_domain or final_domain in allowed_domains):
        error_message = (
            f"Redirect URL '{final_url}' is not allowed. "
            f"Must be within '{app_domain}' or one of {allowed_domains}."
        )
        logger.warning(f"Validation failed: {error_message}")
        raise ValueError(error_message)
    logger.info(f"Validation successful for final_url='{final_url}'")


def generate_redirect_link(app_url: str, redirect_param: str) -> str:
    """
    Generates a redirect URL based on a user-provided 'next' parameter.

    Args:
        app_url: The base URL of the application.
        redirect_param: The URL path or full URL for redirection.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If the final URL is not within the application's domain
                    or an allowed external domain.
    """
    logger.info(
        f"Generating redirect link for app_url='{app_url}', redirect_param='{redirect_param}'"
    )
    parsed_app_url = urlparse.urlparse(app_url)
    app_domain = parsed_app_url.netloc
    logger.debug(f"Application domain parsed as: '{app_domain}'")

    final_url, parsed_final_url = _construct_final_url(app_url, redirect_param)
    logger.debug(f"Constructed final_url='{final_url}', parsed_final_url={parsed_final_url}")

    _validate_redirect_url(final_url, parsed_final_url, app_domain, APPROVED_DOMAINS)
    logger.info(f"Successfully generated and validated redirect_link='{final_url}'")

    return final_url
