import urllib.parse
import logging

# Configure basic logging
# In a real application, this would likely be configured more centrally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _build_base_callback_url(root_url: str, callback_uri: str) -> str:
    """Constructs the base callback URL by joining root_url and callback_uri."""
    logger.debug(f"Building base callback URL with root_url='{root_url}' and callback_uri='{callback_uri}'")
    if root_url.endswith('/') and callback_uri.startswith('/'):
        base_url = root_url + callback_uri[1:]
    elif not root_url.endswith('/') and not callback_uri.startswith('/'):
        # Ensure there's a slash if callback_uri is not an absolute path segment
        if callback_uri:  # Avoid adding slash if callback_uri is empty
            base_url = root_url + '/' + callback_uri
        else:
            base_url = root_url
    else:
        base_url = root_url + callback_uri
    logger.debug(f"Constructed base_url: '{base_url}'")
    return base_url


def _validate_callback_domain(base_url: str, allowed_domains: set) -> str:
    """
    Validates if the domain of the base_url is in the allowed_domains.

    Args:
        base_url: The URL string to validate.
        allowed_domains: A set of allowed domain strings.

    Returns:
        The base_url if its domain is allowed.

    Raises:
        ValueError: If the domain is not in allowed_domains.
    """
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc
    logger.debug(f"Validating domain: '{domain}' from base_url: '{base_url}'")

    if domain not in allowed_domains:
        error_message = (
            f"The domain '{domain}' is not an allowed callback domain. "
            f"Allowed domains are: {', '.join(allowed_domains)}"
        )
        logger.error(f"Domain validation failed for '{base_url}': {error_message}")
        raise ValueError(error_message)
    
    logger.debug(f"Domain '{domain}' is valid.")
    return base_url


def _add_state_to_url(url_str: str, session_state: str) -> str:
    """Adds the session_state as a 'state' query parameter to the URL."""
    logger.debug(f"Adding state='{session_state}' to URL: '{url_str}'")
    parsed_url = urllib.parse.urlparse(url_str)
    query_params = urllib.parse.urlencode({'state': session_state})
    url_with_state = urllib.parse.urlunparse(parsed_url._replace(query=query_params))
    logger.debug(f"URL with state: '{url_with_state}'")
    return url_with_state


def construct_oauth_callback_uri(root_url: str, callback_uri: str, session_state: str) -> str:
    """
    Constructs an OAuth callback URL used during third-party authentication.

    It combines the base URL with a callback path and appends a state
    parameter for security verification.

    Args:
        root_url: The application's main base URL.
        callback_uri: The specific path for the OAuth callback endpoint.
        session_state: A unique value for ensuring the integrity of the OAuth request flow.

    Returns:
        A string representing the finalized OAuth callback URL.

    Raises:
        ValueError: If the computed callback URL's domain is not within
                    the list of allowed domains.
    """
    logger.info(
        f"Constructing OAuth callback URI with root_url='{root_url}', "
        f"callback_uri='{callback_uri}', session_state='{session_state}'"
    )
    try:
        base_url = _build_base_callback_url(root_url, callback_uri)
        logger.info(f"Step 1: Base callback URL constructed: '{base_url}'")

        validated_base_url = _validate_callback_domain(base_url, ALLOWED_CALLBACK_DOMAINS)
        logger.info(f"Step 2: Callback domain validated for: '{validated_base_url}'")

        final_url = _add_state_to_url(validated_base_url, session_state)
        logger.info(f"Step 3: Final OAuth callback URI constructed: '{final_url}'")
        
        return final_url
    except ValueError as e:
        logger.error(f"Error during OAuth callback URI construction: {e}")
        raise  # Re-raise the caught exception
    except Exception as e:
        logger.exception(f"An unexpected error occurred during OAuth callback URI construction: {e}")
        raise # Re-raise the caught exception
