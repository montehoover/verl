import logging
from urllib import parse as urlparse

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

def construct_full_url(base_url, redirect_param):
    """Construct the full redirect URL from base URL and redirect parameter."""
    full_url = urlparse.urljoin(base_url, redirect_param)
    logger.debug(f"Constructed full URL: {full_url} from base: {base_url} and param: {redirect_param}")
    return full_url

def is_same_domain(base_url, redirect_url):
    """Check if the redirect URL is within the same domain as the base URL."""
    parsed_base = urlparse.urlparse(base_url)
    parsed_redirect = urlparse.urlparse(redirect_url)
    same_domain = parsed_redirect.netloc == parsed_base.netloc
    logger.debug(f"Same domain check - Base domain: {parsed_base.netloc}, Redirect domain: {parsed_redirect.netloc}, Result: {same_domain}")
    return same_domain

def is_allowed_external_domain(redirect_url):
    """Check if the redirect URL is in the list of allowed external domains."""
    parsed_redirect = urlparse.urlparse(redirect_url)
    is_allowed = parsed_redirect.netloc in ALLOWED_DOMAINS
    logger.debug(f"Allowed external domain check - Domain: {parsed_redirect.netloc}, Is allowed: {is_allowed}")
    return is_allowed

def validate_redirect_url(base_url, redirect_url):
    """Validate that the redirect URL is either same domain or allowed external."""
    same_domain = is_same_domain(base_url, redirect_url)
    allowed_external = is_allowed_external_domain(redirect_url)
    is_valid = same_domain or allowed_external
    logger.debug(f"URL validation - Same domain: {same_domain}, Allowed external: {allowed_external}, Valid: {is_valid}")
    return is_valid

def generate_redirect_link(app_base_url, redirect_param):
    logger.info(f"Generating redirect link - Base URL: {app_base_url}, Redirect param: {redirect_param}")
    
    try:
        # Construct the full redirect URL
        full_redirect_url = construct_full_url(app_base_url, redirect_param)
        
        # Validate the redirect URL
        if validate_redirect_url(app_base_url, full_redirect_url):
            logger.info(f"Successfully generated redirect URL: {full_redirect_url}")
            return full_redirect_url
        
        # If validation fails, raise ValueError
        error_msg = "Redirect URL is not within the base domain or allowed external domains"
        logger.error(f"Validation failed: {error_msg} - URL: {full_redirect_url}")
        raise ValueError(error_msg)
    
    except Exception as e:
        logger.error(f"Error generating redirect link: {str(e)}", exc_info=True)
        raise
