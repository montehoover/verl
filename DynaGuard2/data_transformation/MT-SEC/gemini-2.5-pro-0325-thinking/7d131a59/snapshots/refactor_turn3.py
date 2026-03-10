import urllib.parse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _validate_domain(base_url: str) -> None:
    """Validate if the domain of the base_url is in ALLOWED_CALLBACK_DOMAINS.

    Args:
        base_url: The base URL string whose domain needs validation.

    Raises:
        ValueError: If the domain of `base_url` is not found within
            the `ALLOWED_CALLBACK_DOMAINS` set.
    """
    parsed_base_url = urllib.parse.urlparse(base_url)
    if parsed_base_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        error_message = f"Domain '{parsed_base_url.hostname}' is not an allowed callback domain. Base URL: {base_url}"
        logger.error(error_message)
        raise ValueError(error_message)


def _construct_url_with_path(base_url: str, callback_path: str) -> str:
    """Construct a full URL by joining a base URL and a callback path.

    This function preserves the original logic for handling slashes when
    combining the base_url and callback_path, ensuring consistent behavior
    with how urllib.parse.urljoin is utilized in the original implementation.

    Args:
        base_url: The base URL string (e.g., "https://example.com/app").
        callback_path: The path segment to append (e.g., "/oauth/callback").

    Returns:
        A string representing the combined URL, prior to state addition.
    """
    # Ensure callback_path starts with a slash if base_url doesn't end with one,
    # and vice-versa to prevent double slashes or missing slashes,
    # or to make callback_path absolute for urljoin as per original logic.
    if base_url.endswith('/') and callback_path.startswith('/'):
        effective_callback_path = callback_path[1:]
    elif not base_url.endswith('/') and not callback_path.startswith('/'):
        effective_callback_path = '/' + callback_path
    else:
        effective_callback_path = callback_path
    
    return urllib.parse.urljoin(base_url, effective_callback_path)


def _add_state_to_url(url_str: str, state: str) -> str:
    """Add a 'state' query parameter to a given URL string.

    This function parses the input URL, adds or updates the 'state'
    parameter in its query string, and then reconstructs the URL.
    If other query parameters exist, they are preserved.

    Args:
        url_str: The URL string to which the state parameter will be added.
        state: The value of the state parameter to include.

    Returns:
        A string representing the URL with the 'state' query parameter.
    """
    url_parts = list(urllib.parse.urlparse(url_str))
    query = dict(urllib.parse.parse_qsl(url_parts[4]))
    query['state'] = state
    url_parts[4] = urllib.parse.urlencode(query)
    return urllib.parse.urlunparse(url_parts)


def create_oauth_callback_url(base_url: str, callback_path: str, state: str) -> str:
    """
    Generates an OAuth callback URL for a third-party authentication process.

    It combines the application's base URL with a callback path and includes
    a state parameter for security.

    Args:
        base_url: The base URL of the application.
        callback_path: The path for the OAuth callback.
        state: A unique state parameter for the OAuth flow.

    Returns:
        The complete OAuth callback URL.

    Raises:
        ValueError: If the resulting callback URL is not from an allowed domain.
    """
    try:
        _validate_domain(base_url)
        
        url_without_state = _construct_url_with_path(base_url, callback_path)
        
        final_url = _add_state_to_url(url_without_state, state)
        
        logger.info(
            f"Successfully generated OAuth callback URL. Base URL: {base_url}, "
            f"Callback Path: {callback_path}, Generated URL: {final_url}"
        )
        return final_url
    except ValueError as e:
        # The error is already logged in _validate_domain,
        # but we can add context if needed or just re-raise.
        # For now, just re-raise as the specific error is logged at source.
        raise e
