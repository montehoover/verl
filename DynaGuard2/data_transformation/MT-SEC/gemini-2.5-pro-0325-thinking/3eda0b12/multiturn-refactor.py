import logging
from urllib import parse as urlparse

# Configure basic logging
# In a real application, this would likely be configured globally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PERMITTED_DOMAINS = ['trusted-site.com', 'partner-domain.org']


def _build_final_url(main_url: str, target_param: str) -> str:
    """
    Constructs the final redirect URL.
    """
    logger.info(f"Building final URL with main_url='{main_url}' and target_param='{target_param}'")
    # Ensure target_param is treated as a string
    final_url = urlparse.urljoin(main_url, str(target_param))
    logger.info(f"Constructed final_url: '{final_url}'")
    return final_url


def _validate_redirect_url(final_url: str, main_domain: str, allowed_domains: list[str]) -> None:
    """
    Validates the final redirect URL against scheme and domain rules.

    Raises:
        ValueError: If the URL scheme is invalid or the domain is not allowed.
    """
    logger.info(f"Validating final_url='{final_url}' against main_domain='{main_domain}' and allowed_domains={allowed_domains}")
    parsed_final_url = urlparse.urlparse(final_url)
    final_domain = parsed_final_url.netloc
    logger.info(f"Parsed final_url: scheme='{parsed_final_url.scheme}', domain='{final_domain}'")

    # Check if the scheme is http or https
    if parsed_final_url.scheme not in ('http', 'https'):
        error_message = "Invalid URL scheme: URL must be http or https."
        logger.error(f"{error_message} (URL: '{final_url}')")
        raise ValueError(error_message)
    logger.info(f"URL scheme '{parsed_final_url.scheme}' is valid.")

    # Check if the final URL is within the main application domain or a permitted domain
    if not (final_domain == main_domain or final_domain in allowed_domains):
        error_message = f"Invalid redirect URL: '{final_url}'. Domain '{final_domain}' is not allowed."
        logger.error(error_message)
        raise ValueError(error_message)
    logger.info(f"URL domain '{final_domain}' is valid.")


def construct_redirect_url(main_url: str, target_param: str) -> str:
    """
    Generates a redirect URL based on a user-provided 'next' parameter.

    Args:
        main_url: The base URL of the application.
        target_param: The URL path or full URL for redirection.

    Returns:
        The final redirect URL as a string.

    Raises:
        ValueError: If the final URL is not within the application's domain
                    or an allowed external domain.
    """
    logger.info(f"Constructing redirect URL for main_url='{main_url}', target_param='{target_param}'")
    parsed_main_url = urlparse.urlparse(main_url)
    main_domain = parsed_main_url.netloc
    logger.info(f"Derived main_domain: '{main_domain}'")

    logger.info("Attempting to build final URL.")
    final_url = _build_final_url(main_url, target_param)
    
    logger.info(f"Attempting to validate final_url: '{final_url}'")
    _validate_redirect_url(final_url, main_domain, PERMITTED_DOMAINS)
    
    logger.info(f"Successfully constructed and validated redirect URL: '{final_url}'")
    return final_url
