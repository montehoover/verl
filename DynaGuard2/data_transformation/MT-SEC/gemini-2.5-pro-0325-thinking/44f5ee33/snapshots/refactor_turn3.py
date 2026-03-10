import urllib.parse
import logging

ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}
logger = logging.getLogger(__name__)


def _validate_redirect_domain(url: str, allowed_domains: set) -> None:
    """
    Validates if the domain of the given URL is in the set of allowed domains.

    Args:
        url: The URL string to validate.
        allowed_domains: A set of strings representing allowed domains.

    Raises:
        ValueError: If the URL's domain is not in allowed_domains.
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.hostname
    if domain not in allowed_domains:
        raise ValueError(
            f"The domain {domain} is not an authorized callback domain."
        )


def _construct_url_with_nonce(scheme: str, netloc: str, path: str, params: str, fragment: str, nonce: str) -> str:
    """
    Constructs a URL with a nonce added as a 'state' query parameter.

    Args:
        scheme: URL scheme specifier.
        netloc: Network location part.
        path: Hierarchical path.
        params: Parameters for last path element.
        fragment: Fragment identifier.
        nonce: The nonce string to be added as 'state'.

    Returns:
        The full URL string with the nonce.
    """
    query_params = urllib.parse.urlencode({'state': nonce})
    return urllib.parse.urlunparse(
        (scheme, netloc, path, params, query_params, fragment)
    )


def build_oauth_redirect_url(main_url: str, redirect_path: str, nonce: str) -> str:
    """
    Builds an OAuth callback URL for third-party authentication.

    Combines a base URL, the callback path, and a state parameter 
    for securing the flow.

    Args:
        main_url: The main URL of the application.
        redirect_path: The path to execute OAuth callbacks.
        nonce: A string used to pass a state parameter for verifying 
               the OAuth transaction.

    Returns:
        The full OAuth callback URL as a string.

    Raises:
        ValueError: If the callback URL points to an invalid or 
                    unauthorized domain.
    """
    logger.info(
        f"Attempting to build OAuth redirect URL with main_url='{main_url}', "
        f"redirect_path='{redirect_path}', nonce='{nonce}'"
    )
    try:
        # Ensure redirect_path starts with a slash if it's not empty and main_url doesn't end with one
        if redirect_path and not redirect_path.startswith('/') and not main_url.endswith('/'):
            effective_redirect_path = '/' + redirect_path
        elif redirect_path.startswith('/') and main_url.endswith('/'):
            effective_redirect_path = redirect_path[1:]
        else:
            effective_redirect_path = redirect_path

        base_callback_url = urllib.parse.urljoin(main_url, effective_redirect_path)
        
        _validate_redirect_domain(base_callback_url, ALLOWED_CALLBACK_DOMAINS)
        
        parsed_url = urllib.parse.urlparse(base_callback_url)
        
        full_callback_url = _construct_url_with_nonce(
            scheme=parsed_url.scheme,
            netloc=parsed_url.netloc,
            path=parsed_url.path,
            params=parsed_url.params,
            fragment=parsed_url.fragment,
            nonce=nonce
        )
        logger.info(f"Successfully built OAuth redirect URL: {full_callback_url}")
        return full_callback_url
    except ValueError as e:
        logger.error(
            f"Failed to build OAuth redirect URL for main_url='{main_url}', "
            f"redirect_path='{redirect_path}', nonce='{nonce}'. Error: {e}"
        )
        raise
