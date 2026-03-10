import urllib.parse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _validate_callback_domain(root_url: str) -> None:
    """
    Validates if the domain of the root_url is in ALLOWED_CALLBACK_DOMAINS.

    Args:
        root_url: The main URL of the application.

    Raises:
        ValueError: If the domain is not allowed.
    """
    parsed_root_url = urllib.parse.urlparse(root_url)
    if parsed_root_url.hostname not in ALLOWED_CALLBACK_DOMAINS:
        error_message = f"Invalid callback domain: {parsed_root_url.hostname}. Not in allowed list: {ALLOWED_CALLBACK_DOMAINS}"
        logger.error(error_message)
        raise ValueError(error_message)


def _build_base_url_with_path(root_url: str, path_for_callback: str) -> str:
    """
    Constructs the base callback URL by joining root_url and path_for_callback.

    Args:
        root_url: The main URL of the application.
        path_for_callback: The path to execute OAuth callbacks.

    Returns:
        The combined base URL string.
    """
    # Ensure root_url ends with a slash for correct urljoin behavior with relative paths
    if not root_url.endswith('/'):
        root_url += '/'
    
    # urllib.parse.urljoin correctly handles joining, including leading/trailing slashes.
    return urllib.parse.urljoin(root_url, path_for_callback.lstrip('/'))


def _append_state_query_parameter(base_url: str, session_token: str) -> str:
    """
    Appends the session_token as a 'state' query parameter to the base_url.

    Args:
        base_url: The base URL to append the query parameter to.
        session_token: The state token.

    Returns:
        The URL string with the state query parameter.
    """
    params = {'state': session_token}
    query_string = urllib.parse.urlencode(params)
    return f"{base_url}?{query_string}"


def assemble_oauth_callback_url(root_url: str, path_for_callback: str, session_token: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication.

    Args:
        root_url: The main URL of the application.
        path_for_callback: The path to execute OAuth callbacks.
        session_token: A string used to pass a state parameter for verifying the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or unauthorized domain.
    """
    # Step 1: Validate the domain
    _validate_callback_domain(root_url)

    # Step 2: Construct the base URL with the callback path
    base_url = _build_base_url_with_path(root_url, path_for_callback)

    # Step 3: Append the state query parameter
    final_url = _append_state_query_parameter(base_url, session_token)
    
    logger.info(f"Successfully assembled OAuth callback URL: {final_url}")
    return final_url
