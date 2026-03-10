import urllib.parse
import logging

# Configure logging
# In a real application, this would likely be configured globally.
# For this example, basic configuration is sufficient.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _construct_base_callback_url(application_url: str, callback_route: str) -> str:
    """Constructs the base callback URL from the application URL and callback route."""
    logger.debug(f"Constructing base callback URL with app_url='{application_url}', route='{callback_route}'")
    if callback_route and not callback_route.startswith('/') and not application_url.endswith('/'):
        effective_callback_route = '/' + callback_route
    elif callback_route.startswith('/') and application_url.endswith('/'):
        effective_callback_route = callback_route[1:]
    else:
        effective_callback_route = callback_route
    base_url = urllib.parse.urljoin(application_url, effective_callback_route)
    logger.debug(f"Constructed base callback URL: '{base_url}'")
    return base_url


def _validate_callback_domain(base_callback_url: str, application_url: str) -> urllib.parse.ParseResult:
    """
    Validates the domain of the base callback URL.

    Args:
        base_callback_url: The constructed base callback URL.
        application_url: The root URL of the application.

    Returns:
        The parsed callback URL (urllib.parse.ParseResult) if valid.

    Raises:
        ValueError: If the callback URL domain is not allowed.
    """
    logger.debug(f"Validating callback domain for base_url='{base_callback_url}', app_url='{application_url}'")
    parsed_app_url = urllib.parse.urlparse(application_url)
    app_domain = parsed_app_url.netloc.split(':')[0]  # Remove port if present

    parsed_callback_url = urllib.parse.urlparse(base_callback_url)
    callback_domain = parsed_callback_url.netloc.split(':')[0]  # Remove port if present

    if not (callback_domain == app_domain or callback_domain in ALLOWED_CALLBACK_DOMAINS):
        error_message = (
            f"Callback URL domain '{callback_domain}' is not allowed. "
            f"Must be '{app_domain}' or one of {ALLOWED_CALLBACK_DOMAINS}."
        )
        logger.error(f"Domain validation failed: {error_message}")
        raise ValueError(error_message)
    
    logger.debug(f"Callback domain '{callback_domain}' validated successfully.")
    return parsed_callback_url


def _append_token_state_to_url(parsed_callback_url: urllib.parse.ParseResult, token_state: str) -> str:
    """Appends the token state as a query parameter to the parsed callback URL."""
    logger.debug(f"Appending token_state='{token_state}' to URL='{parsed_callback_url.geturl()}'")
    query_params = {'state': token_state}
    final_url = urllib.parse.urlunparse(
        parsed_callback_url._replace(query=urllib.parse.urlencode(query_params))
    )
    logger.debug(f"URL with token state: '{final_url}'")
    return final_url


def assemble_oauth_callback(application_url: str, callback_route: str, token_state: str) -> str:
    """
    Creates an OAuth callback URL typically used in third-party authentication workflows.
    It constructs the URL by merging the base application URL with a callback path
    and appends a security-related state parameter.

    Args:
        application_url: A string identifying the root URL for the application.
        callback_route: A string representing the endpoint for the OAuth callback handler.
        token_state: A string that provides a unique identifier to maintain the
                     integrity of the OAuth exchange.

    Returns:
        A string that contains the fully assembled OAuth callback URL.

    Raises:
        ValueError: Raised when the callback URL domain fails to meet the security
                    requirements of allowed or same-domain restrictions.
    """
    logger.info(
        f"Assembling OAuth callback URL with application_url='{application_url}', "
        f"callback_route='{callback_route}', token_state='{token_state}'"
    )
    try:
        base_callback_url = _construct_base_callback_url(application_url, callback_route)
        parsed_validated_callback_url = _validate_callback_domain(base_callback_url, application_url)
        final_url = _append_token_state_to_url(parsed_validated_callback_url, token_state)
        logger.info(f"Successfully assembled OAuth callback URL: '{final_url}'")
        return final_url
    except ValueError as e:
        logger.error(f"Error during OAuth callback URL assembly: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred during OAuth callback URL assembly: {e}", exc_info=True)
        raise
