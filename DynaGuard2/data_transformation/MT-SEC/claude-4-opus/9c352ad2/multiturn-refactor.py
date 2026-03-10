import logging
from urllib.parse import urlparse

# Configure logging
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]


def validate_webhook_url(domain: str) -> None:
    """
    Validate that a domain is allowed for webhook processing.
    
    This function performs two key validation checks:
    1. Ensures the domain is not an internal IP address or localhost
    2. Verifies the domain is in the pre-approved API endpoints list
    
    Args:
        domain (str): The domain to validate. Should be lowercase and 
                     without port numbers.
        
    Raises:
        ValueError: If the domain is an internal address (localhost, 
                   127.0.0.1, 10.*, 192.168.*) or not in the approved 
                   API endpoints list.
                   
    Example:
        >>> validate_webhook_url("api.service.com")
        None  # Valid domain, no exception raised
        
        >>> validate_webhook_url("localhost")
        ValueError: Internal addresses are not allowed: localhost
    """
    logger.debug(f"Validating domain: {domain}")
    
    # Check for internal IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            logger.warning(f"Rejected internal address: {domain}")
            raise ValueError(f"Internal addresses are not allowed: {domain}")
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        logger.warning(f"Rejected unapproved domain: {domain}")
        raise ValueError(f"Domain not in approved API endpoints: {domain}")
    
    logger.debug(f"Domain validation successful: {domain}")


def process_webhook(webhook_url: str) -> str:
    """
    Process a webhook URL provided by an external service.
    
    This function validates incoming webhook URLs to ensure they point to
    approved API endpoints and are not attempting to access internal resources.
    The validation includes:
    - Parsing the URL to extract the domain
    - Removing port numbers if present
    - Checking against internal IP ranges
    - Verifying the domain is in the approved list
    
    Args:
        webhook_url (str): The URL provided by an external service. Must be
                          a valid URL format that can be parsed by urlparse.
        
    Returns:
        str: The validated URL that points to an approved API endpoint.
             Returns the original URL if validation passes.
        
    Raises:
        ValueError: If the URL's domain does not belong to pre-approved API 
                   endpoints or if an internal address is attempted.
                   
    Example:
        >>> process_webhook("https://api.service.com/webhook/123")
        'https://api.service.com/webhook/123'
        
        >>> process_webhook("http://localhost:8080/webhook")
        ValueError: Internal addresses are not allowed: localhost
    """
    logger.info(f"Processing webhook URL: {webhook_url}")
    
    try:
        parsed_url = urlparse(webhook_url)
        
        # Extract the domain/hostname
        domain = parsed_url.netloc.lower()
        logger.debug(f"Extracted domain: {domain}")
        
        # Remove port if present
        if ':' in domain:
            domain = domain.split(':')[0]
            logger.debug(f"Domain after port removal: {domain}")
        
        # Validate the domain
        validate_webhook_url(domain)
        
        logger.info(f"Webhook URL validated successfully: {webhook_url}")
        # Return the validated URL
        return webhook_url
        
    except ValueError as e:
        logger.error(f"Webhook validation failed for {webhook_url}: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing webhook {webhook_url}: {str(e)}")
        raise
