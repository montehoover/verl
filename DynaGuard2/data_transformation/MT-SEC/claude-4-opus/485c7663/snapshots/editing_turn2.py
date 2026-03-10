from datetime import datetime
import logging
from urllib.parse import urlparse

ALLOWED_SCHEMES = ["https"]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def enforce_api_security(api_endpoint: str) -> str:
    """
    Logs API endpoint access for audit purposes and ensures secure HTTPS access.
    
    Args:
        api_endpoint: The URL that the application is attempting to access
        
    Returns:
        The original URL string for further processing
        
    Raises:
        ValueError: If the endpoint doesn't use HTTPS scheme
    """
    # Parse the URL to check the scheme
    parsed_url = urlparse(api_endpoint)
    
    # Check if the scheme is in allowed schemes (https only)
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"API endpoint '{api_endpoint}' must use HTTPS for secure access")
    
    # Log the safe API endpoint access with timestamp
    logging.info(f"API endpoint accessed: {api_endpoint}")
    
    return api_endpoint
