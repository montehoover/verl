from urllib.parse import urlparse
import logging

# Configure logger
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def is_internal_network(domain):
    """
    Check if a domain points to an internal network location.
    
    Args:
        domain: hostname/domain to check
        
    Returns:
        True if domain is internal, False otherwise
    """
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False

def is_approved_domain(domain):
    """
    Check if a domain is in the approved API endpoints list.
    
    Args:
        domain: hostname/domain to check
        
    Returns:
        True if domain is approved, False otherwise
    """
    return domain in APPROVED_API_DOMAINS

def process_hook(hook_url):
    """
    Handles a webhook URL sent by an external service.
    
    Args:
        hook_url: a string representing the URL provided by the external service
        
    Returns:
        A safely validated URL that references an approved API endpoint
        
    Raises:
        ValueError: if the URL's domain is outside the list of approved API endpoints
                   or aims at an internal network location
    """
    logger.info(f"Processing webhook URL: {hook_url}")
    
    # Parse the URL
    parsed = urlparse(hook_url)
    logger.debug(f"Parsed URL - scheme: {parsed.scheme}, domain: {parsed.hostname}, path: {parsed.path}")
    
    # Extract the domain/hostname
    domain = parsed.hostname
    
    if not domain:
        logger.error(f"Invalid URL: no domain found in {hook_url}")
        raise ValueError("Invalid URL: no domain found")
    
    # Check for internal network locations
    logger.debug(f"Checking if domain {domain} is internal network")
    if is_internal_network(domain):
        logger.warning(f"Rejected URL - points to internal network location: {domain}")
        raise ValueError(f"URL points to internal network location: {domain}")
    logger.debug(f"Domain {domain} passed internal network check")
    
    # Check if domain is in approved list
    logger.debug(f"Checking if domain {domain} is in approved list")
    if not is_approved_domain(domain):
        logger.warning(f"Rejected URL - domain not in approved API endpoints: {domain}")
        raise ValueError(f"Domain not in approved API endpoints: {domain}")
    logger.debug(f"Domain {domain} is approved")
    
    # Return the validated URL
    logger.info(f"Successfully validated webhook URL: {hook_url}")
    return hook_url
