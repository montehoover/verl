import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_RETURN_DOMAINS = {'app.example.com', 'secure.example.org', 'dashboard.example.net'}


def _extract_domain(url):
    """Extract the domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc


def _is_domain_allowed(domain, allowed_domains):
    """Check if a domain is in the allowed list."""
    return domain in allowed_domains


def _validate_domain(domain, allowed_domains):
    """Validate that a domain is in the allowed list."""
    if not _is_domain_allowed(domain, allowed_domains):
        logger.error(f"Domain validation failed: '{domain}' is not in the allowed domains list")
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    logger.debug(f"Domain validation successful: '{domain}' is allowed")


def _merge_query_parameters(existing_params, new_params):
    """Merge new query parameters with existing ones."""
    merged_params = existing_params.copy()
    merged_params.update(new_params)
    return merged_params


def _build_url_with_params(parsed_url, query_params):
    """Build a URL from parsed components with updated query parameters."""
    new_query = urllib.parse.urlencode(query_params, doseq=True)
    return urllib.parse.urlunparse((
        parsed_url.scheme,
        parsed_url.netloc,
        parsed_url.path,
        parsed_url.params,
        new_query,
        parsed_url.fragment
    ))


def create_sso_url(auth_endpoint, uid, redirect_url):
    """
    Constructs an SSO redirect URL that facilitates authentication with an external identity provider.
    
    Args:
        auth_endpoint: A string indicating the SSO endpoint URL.
        uid: A string representing the unique user ID.
        redirect_url: A string pointing to the return URL after SSO completes.
    
    Returns:
        A string containing the full SSO redirect URL.
    
    Raises:
        ValueError: Raised if the redirect_url's domain is not part of the approved domains list.
    """
    logger.info(f"Starting SSO URL construction for user: {uid}")
    
    # Validate the redirect URL domain
    redirect_domain = _extract_domain(redirect_url)
    logger.debug(f"Extracted domain from redirect URL: {redirect_domain}")
    _validate_domain(redirect_domain, ALLOWED_RETURN_DOMAINS)
    
    # Prepare SSO parameters
    sso_params = {
        'uid': uid,
        'redirect': redirect_url
    }
    logger.debug(f"Prepared SSO parameters: {sso_params}")
    
    # Parse the auth endpoint and merge parameters
    parsed_endpoint = urllib.parse.urlparse(auth_endpoint)
    logger.debug(f"Parsed auth endpoint: {auth_endpoint}")
    
    existing_params = urllib.parse.parse_qs(parsed_endpoint.query)
    merged_params = _merge_query_parameters(existing_params, sso_params)
    logger.debug(f"Merged query parameters: {merged_params}")
    
    # Build and return the final SSO URL
    sso_url = _build_url_with_params(parsed_endpoint, merged_params)
    logger.info(f"Successfully constructed SSO URL for user: {uid}")
    logger.debug(f"Final SSO URL: {sso_url}")
    
    return sso_url
