from urllib.parse import urlparse
import logging

# Configure logger
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_domain(url):
    """
    Extracts the domain from a URL, removing any port information.
    
    This function parses the given URL and extracts the network location (domain/hostname),
    converting it to lowercase for consistent comparison. If a port is included in the
    domain (e.g., "example.com:8080"), it will be removed.
    
    Args:
        url (str): A string representing a URL from which to extract the domain.
                   Expected format: scheme://domain[:port]/path
        
    Returns:
        str: The domain/hostname in lowercase without port information.
             For example, "https://API.Service.COM:443/path" returns "api.service.com"
        
    Examples:
        >>> extract_domain("https://api.service.com:8080/webhook")
        'api.service.com'
        >>> extract_domain("http://localhost:3000")
        'localhost'
    """
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.lower()
    
    # Remove port if present
    if ':' in domain:
        domain = domain.split(':')[0]
    
    return domain

def is_approved_domain(domain):
    """
    Checks if a domain is in the approved API endpoint list.
    
    This function performs a simple membership check against the predefined
    APPROVED_API_DOMAINS list to determine if the given domain is authorized
    for webhook interactions.
    
    Args:
        domain (str): A string representing the domain to check.
                      Should be lowercase and without port information.
        
    Returns:
        bool: True if the domain is in the approved list, False otherwise.
        
    Examples:
        >>> is_approved_domain("api.service.com")
        True
        >>> is_approved_domain("malicious.site.com")
        False
    """
    return domain in APPROVED_API_DOMAINS

def is_internal_address(domain):
    """
    Checks if a domain represents an internal/private network address.
    
    This function checks if the given domain matches any known internal or
    private network address patterns, including localhost, loopback addresses,
    and private IP ranges. This helps prevent SSRF (Server-Side Request Forgery)
    attacks by blocking access to internal resources.
    
    Args:
        domain (str): A string representing the domain to check.
                      Should be lowercase and without port information.
        
    Returns:
        bool: True if the domain matches any internal address pattern, False otherwise.
        
    Examples:
        >>> is_internal_address("localhost")
        True
        >>> is_internal_address("192.168.1.100")
        True
        >>> is_internal_address("api.service.com")
        False
    """
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            return True
    return False

def validate_webhook(webhook_link):
    """
    Validates a webhook URL from an external source and returns a secure URL for internal API calls.
    
    This function performs comprehensive validation on the provided webhook URL to ensure
    it meets security requirements. It extracts the domain, verifies it against an approved
    whitelist, and ensures it doesn't point to internal network addresses. All validation
    steps are logged for debugging and security auditing purposes.
    
    Args:
        webhook_link (str): A string representing the external webhook URL to validate.
                           Expected to be a complete URL with scheme and domain.
        
    Returns:
        str: The original webhook_link if all validations pass. This URL is considered
             safe for internal API interactions.
        
    Raises:
        ValueError: Raised in the following cases:
                   - If the URL domain is not in the APPROVED_API_DOMAINS list
                   - If the URL attempts to access internal/private network addresses
                     (localhost, 127.0.0.1, 10.x.x.x, 192.168.x.x)
                   
    Examples:
        >>> validate_webhook("https://api.service.com/webhook/callback")
        'https://api.service.com/webhook/callback'
        
        >>> validate_webhook("https://malicious.site.com/webhook")
        ValueError: Domain 'malicious.site.com' is not in the approved API endpoint list
        
        >>> validate_webhook("http://localhost:8080/internal")
        ValueError: URL cannot access internal/private network addresses
    """
    logger.info(f"Starting webhook validation for URL: {webhook_link}")
    
    # Extract the domain from the URL
    domain = extract_domain(webhook_link)
    logger.debug(f"Extracted domain: {domain}")
    
    # Check if the domain is in the approved list
    if not is_approved_domain(domain):
        logger.warning(f"Validation failed: Domain '{domain}' is not approved. URL: {webhook_link}")
        raise ValueError(f"Domain '{domain}' is not in the approved API endpoint list")
    
    logger.debug(f"Domain '{domain}' is approved")
    
    # Check if the URL is trying to access internal/private network addresses
    if is_internal_address(domain):
        logger.warning(f"Validation failed: Domain '{domain}' is an internal address. URL: {webhook_link}")
        raise ValueError(f"URL cannot access internal/private network addresses")
    
    logger.debug(f"Domain '{domain}' is not an internal address")
    logger.info(f"Webhook validation successful for URL: {webhook_link}")
    
    # Return the validated URL
    return webhook_link
