import urllib.parse
import logging

# Configure basic logging
# In a real application, this might be configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _validate_redirect_domain(redirect_url: str, allowed_domains: set) -> None:
    """
    Validates if the redirect URL's domain is in the allowed list.

    Args:
        redirect_url: The URL to validate.
        allowed_domains: A set of allowed domain hostnames.

    Raises:
        ValueError: If the redirect_url's domain is not in allowed_domains.
    """
    logger.info(f"Validating redirect URL: {redirect_url}")
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    domain_hostname = parsed_redirect_url.hostname

    if domain_hostname not in allowed_domains:
        error_message = f"Redirect URL domain '{domain_hostname}' is not allowed. Allowed domains: {allowed_domains}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    logger.info(f"Redirect URL domain '{domain_hostname}' is valid.")


def _construct_sso_url_with_params(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Constructs the SSO URL with user ID and redirect URL as query parameters.

    Args:
        auth_endpoint: The base SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL string.
    """
    logger.info(f"Constructing SSO URL for user '{uid}' with endpoint '{auth_endpoint}' and redirect '{redirect_url}'")
    params = {
        'user_id': uid,
        'return_to': redirect_url
    }
    
    # Determine the correct separator based on whether auth_endpoint already has query params
    separator = '&' if '?' in auth_endpoint else '?'
    
    # Append query parameters to the auth_endpoint
    query_string = urllib.parse.urlencode(params)
    sso_url = f"{auth_endpoint}{separator}{query_string}"
    
    logger.info(f"Successfully constructed SSO URL: {sso_url}")
    return sso_url


def create_sso_url(auth_endpoint: str, uid: str, redirect_url: str) -> str:
    """
    Constructs an SSO redirect URL.

    The URL is crafted by merging the SSO endpoint, user-specific details,
    and the destination return URL.

    Args:
        auth_endpoint: The SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL.

    Raises:
        ValueError: If the redirect_url's domain is not part of the
                    approved domains list.
    """
    logger.info(f"Initiating SSO URL creation for user '{uid}'. Auth endpoint: '{auth_endpoint}', Redirect URL: '{redirect_url}'")
    
    _validate_redirect_domain(redirect_url, ALLOWED_RETURN_DOMAINS)
    
    sso_url = _construct_sso_url_with_params(auth_endpoint, uid, redirect_url)
    
    logger.info(f"SSO URL creation successful for user '{uid}'.")
    return sso_url
