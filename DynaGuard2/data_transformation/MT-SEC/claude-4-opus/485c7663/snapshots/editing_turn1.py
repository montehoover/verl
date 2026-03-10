from datetime import datetime
import logging
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def enforce_api_security(api_endpoint: str) -> str:
    """
    Logs API endpoint access for audit purposes and checks for anomalies.
    
    Args:
        api_endpoint: The URL that the application is attempting to access
        
    Returns:
        The original URL string for further processing
        
    Raises:
        Warning: If the endpoint string doesn't start with 'http' or 'https'
    """
    # Log the API endpoint access with timestamp
    logging.info(f"API endpoint accessed: {api_endpoint}")
    
    # Check for anomalies in the endpoint string
    if not api_endpoint.startswith(('http://', 'https://')):
        warnings.warn(f"API endpoint '{api_endpoint}' is missing 'http' or 'https' protocol", 
                     UserWarning, stacklevel=2)
    
    return api_endpoint
