import logging
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def is_domain_valid(domain: str) -> bool:
    """
    Check if a domain is in the list of approved domains.
    
    Args:
        domain: The domain to validate
        
    Returns:
        True if the domain is valid, False otherwise
    """
    # Normalize domain to lowercase
    normalized_domain = domain.lower()
    
    # Remove www. prefix if present
    if normalized_domain.startswith('www.'):
        normalized_domain = normalized_domain[4:]
    
    return normalized_domain in VALID_REDIRECT_DOMAINS

def secure_user_redirect(url_input: str) -> str:
    """
    Securely process a user-provided URL for redirection.
    
    Args:
        url_input: The URL input provided by the user
        
    Returns:
        A validated and safe-to-use URL for redirection
        
    Raises:
        ValueError: If the domain is not in the approved list
    """
    logger.info(f"URL redirection attempt - Original URL: {url_input}")
    
    parsed_url = urlparse(url_input)
    domain = parsed_url.netloc
    
    if not is_domain_valid(domain):
        logger.warning(f"URL redirection failed - Original URL: {url_input}, Domain: {domain}, Validation Result: REJECTED, Reason: Domain not in approved list")
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    logger.info(f"URL redirection successful - Original URL: {url_input}, Domain: {domain}, Validation Result: APPROVED, Final Redirection URL: {url_input}")
    return url_input
