import logging
from urllib import parse as urlparse

ALLOWED_DOMAINS = ['trusted-site.com', 'partner-domain.org']

# Configure logger
logger = logging.getLogger(__name__)

def parse_redirect_url(domain_base_url, next_redirect_param):
    """Parse and construct the full redirect URL."""
    logger.debug(f"Parsing redirect URL - base: {domain_base_url}, next: {next_redirect_param}")
    
    if next_redirect_param.startswith('/'):
        # Relative path - combine with base URL
        redirect_url = urlparse.urljoin(domain_base_url, next_redirect_param)
        logger.debug(f"Constructed relative path redirect: {redirect_url}")
    else:
        # Absolute URL or full path
        redirect_url = next_redirect_param
        logger.debug(f"Using absolute URL: {redirect_url}")
    
    return redirect_url

def validate_url_scheme(redirect_url):
    """Validate that the URL has an acceptable scheme."""
    parsed = urlparse.urlparse(redirect_url)
    logger.debug(f"Validating URL scheme for: {redirect_url}")
    
    if parsed.scheme and parsed.scheme not in ['http', 'https']:
        logger.error(f"Invalid URL scheme detected: {parsed.scheme}")
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}")
    
    logger.debug(f"URL scheme validation passed: {parsed.scheme or 'none'}")
    return redirect_url

def validate_domain(redirect_url, domain_base_url):
    """Validate that the redirect URL is within base domain or allowed domains."""
    base_parsed = urlparse.urlparse(domain_base_url)
    redirect_parsed = urlparse.urlparse(redirect_url)
    
    logger.debug(f"Validating domain - redirect netloc: {redirect_parsed.netloc or 'none'}, base netloc: {base_parsed.netloc}")
    
    # Check if redirect is within the base domain
    if redirect_parsed.netloc == base_parsed.netloc or not redirect_parsed.netloc:
        logger.info(f"Domain validation passed: redirect within base domain {base_parsed.netloc}")
        return redirect_url
    
    # Check if redirect is to an allowed external domain
    if redirect_parsed.netloc in ALLOWED_DOMAINS:
        logger.info(f"Domain validation passed: {redirect_parsed.netloc} is in allowed domains list")
        return redirect_url
    
    # If neither condition is met, raise ValueError
    logger.error(f"Domain validation failed: {redirect_parsed.netloc} not allowed")
    raise ValueError(f"Redirect URL {redirect_url} is not within the base domain {base_parsed.netloc} or in the allowed domains list")

def construct_redirect_link(domain_base_url, next_redirect_param):
    logger.info(f"Starting redirect link construction - base: {domain_base_url}, next: {next_redirect_param}")
    
    try:
        # Pipeline: parse -> validate scheme -> validate domain
        redirect_url = parse_redirect_url(domain_base_url, next_redirect_param)
        redirect_url = validate_url_scheme(redirect_url)
        redirect_url = validate_domain(redirect_url, domain_base_url)
        
        logger.info(f"Successfully constructed redirect link: {redirect_url}")
        return redirect_url
    except ValueError as e:
        logger.error(f"Failed to construct redirect link: {str(e)}")
        raise
