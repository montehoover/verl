import urllib.parse
import logging

# Configure logging
logger = logging.getLogger(__name__)
# Example basic configuration (can be configured externally in a real app)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


ALLOWED_CALLBACK_DOMAINS = {'auth.myapp.com', 'login.myapp.org', 'oauth.myapp.net'}


def _validate_callback_domain(host_url: str, path_callback: str, allowed_domains: set) -> None:
    """
    Validates the domain of the callback URL.

    Args:
        host_url: The root URL for the application.
        path_callback: The endpoint for the OAuth callback handler.
        allowed_domains: A set of allowed domains for the callback.

    Raises:
        ValueError: If the callback URL domain is not in the allowed_domains.
    """
    parsed_host_url = urllib.parse.urlparse(host_url)
    host_domain = parsed_host_url.hostname

    # Determine the effective domain to check
    domain_to_check = host_domain
    parsed_callback_path = urllib.parse.urlparse(path_callback)

    if parsed_callback_path.hostname: # path_callback is an absolute URL
        domain_to_check = parsed_callback_path.hostname
        if domain_to_check not in allowed_domains:
            raise ValueError(
                f"Callback domain '{domain_to_check}' from path_callback is not an allowed domain."
            )
    elif host_domain not in allowed_domains: # path_callback is relative, check host_domain
        raise ValueError(
            f"Host domain '{host_domain}' for callback is not an allowed domain."
        )


def _build_oauth_url_with_state(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Constructs the OAuth callback URL with the state parameter.

    Args:
        host_url: The root URL for the application.
        path_callback: The endpoint for the OAuth callback handler.
        session_id: The state parameter value.

    Returns:
        The fully assembled OAuth callback URL with the state parameter.
    """
    # Ensure path_callback starts with a slash if it's a relative path and not a full URL
    if not path_callback.startswith('/') and not urllib.parse.urlparse(path_callback).scheme:
        path_callback = '/' + path_callback

    # Construct the base callback URL
    callback_base_url = urllib.parse.urljoin(host_url, path_callback)

    # Add the state parameter
    url_parts = list(urllib.parse.urlparse(callback_base_url))
    query_params = urllib.parse.parse_qs(url_parts[4])
    query_params['state'] = [session_id]
    url_parts[4] = urllib.parse.urlencode(query_params, doseq=True)

    return urllib.parse.urlunparse(url_parts)


def generate_oauth_url(host_url: str, path_callback: str, session_id: str) -> str:
    """
    Creates an OAuth callback URL.

    Constructs the URL by merging the base application URL with a callback path
    and appends a security-related state parameter.

    Args:
        host_url: The root URL for the application.
        path_callback: The endpoint for the OAuth callback handler.
        session_id: A unique identifier to maintain the integrity of the OAuth exchange.

    Returns:
        The fully assembled OAuth callback URL.

    Raises:
        ValueError: If the callback URL domain fails to meet security requirements.
    """
    logger.info(
        f"Generating OAuth URL with host_url='{host_url}', path_callback='{path_callback}', session_id='{session_id}'"
    )
    try:
        _validate_callback_domain(host_url, path_callback, ALLOWED_CALLBACK_DOMAINS)
        oauth_url = _build_oauth_url_with_state(host_url, path_callback, session_id)
        logger.info(f"Successfully generated OAuth URL: {oauth_url}")
        return oauth_url
    except ValueError as e:
        logger.error(f"Failed to generate OAuth URL: {e}")
        raise
