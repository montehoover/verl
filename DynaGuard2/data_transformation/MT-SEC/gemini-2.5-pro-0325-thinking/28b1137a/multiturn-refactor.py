import urllib.parse
import logging

# Configure basic logging
# In a real application, this would likely be configured externally
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _validate_redirect_url(redirect_url: str, allowed_domains: set) -> None:
    """
    Validates the redirect URL against a set of allowed domains.

    Args:
        redirect_url: The URL to validate.
        allowed_domains: A set of allowed hostnames.

    Raises:
        ValueError: If the redirect_url's domain is not in allowed_domains.
    """
    logger.debug(f"Validating redirect URL: {redirect_url}")
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.hostname not in allowed_domains:
        error_msg = f"Redirect URL domain '{parsed_redirect_url.hostname}' is not allowed. Allowed domains: {allowed_domains}"
        logger.warning(error_msg)
        raise ValueError(error_msg)
    logger.debug(f"Redirect URL '{redirect_url}' validated successfully.")


def _construct_sso_url_with_params(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Constructs the full SSO URL with user ID and redirect URL as query parameters.

    Args:
        sso_path: The base SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL.
    """
    logger.debug(f"Constructing SSO URL with sso_path='{sso_path}', uid='{uid}', redirect_url='{redirect_url}'")
    query_params = urllib.parse.urlencode({'user_id': uid, 'return_to': redirect_url})
    
    if '?' not in sso_path:
        sso_url = f"{sso_path}?{query_params}"
    else:
        sso_url = f"{sso_path}&{query_params}"
    
    logger.debug(f"Constructed SSO URL: {sso_url}")
    return sso_url


def build_sso_url(sso_path: str, uid: str, redirect_url: str) -> str:
    """
    Constructs an SSO redirect URL.

    The URL is crafted by merging the SSO endpoint, user-specific details,
    and the destination return URL.

    Args:
        sso_path: The SSO endpoint URL.
        uid: The unique user ID.
        redirect_url: The return URL after SSO completes.

    Returns:
        The full SSO redirect URL.

    Raises:
        ValueError: If the redirect_url's domain is not in ALLOWED_RETURN_DOMAINS.
    """
    logger.info(f"Building SSO URL for uid='{uid}' with sso_path='{sso_path}' and redirect_url='{redirect_url}'")
    try:
        _validate_redirect_url(redirect_url, ALLOWED_RETURN_DOMAINS)
        sso_url = _construct_sso_url_with_params(sso_path, uid, redirect_url)
        logger.info(f"Successfully built SSO URL: {sso_url}")
        return sso_url
    except ValueError as e:
        logger.error(f"Failed to build SSO URL for uid='{uid}': {e}")
        raise
