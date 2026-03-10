import logging
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_domain(url: str) -> str:
    """Extract and normalize the domain from a URL."""
    parsed_url = urlparse(url)
    
    if not parsed_url.scheme:
        raise ValueError("URL must include a scheme (http:// or https://)")
    
    if not parsed_url.netloc:
        raise ValueError("URL must include a domain")
    
    domain = parsed_url.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def is_domain_authorized(domain: str) -> bool:
    """Check if a domain is in the list of authorized redirect domains."""
    return domain in AUTHORIZED_REDIRECT_DOMAINS

def process_url_redirect(input_url: str) -> str:
    logger.info(f"Processing URL redirect request for: {input_url}")
    
    try:
        domain = extract_domain(input_url)
        logger.debug(f"Extracted domain: {domain}")
        
        if not is_domain_authorized(domain):
            error_msg = f"Domain '{domain}' is not in the list of approved domains"
            logger.warning(f"Validation failed for URL '{input_url}': {error_msg}")
            raise ValueError(error_msg)
        
        logger.info(f"URL validation successful for: {input_url} (domain: {domain})")
        return input_url
        
    except ValueError as e:
        logger.error(f"Error processing URL '{input_url}': {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing URL '{input_url}': {str(e)}")
        raise
