from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_callback(callback_url):
    """
    Processes webhook data to extract and log callback URLs.
    
    Args:
        callback_url: A string representing the URL provided in the webhook payload.
    
    Returns:
        The parsed URL as a string.
    
    Raises:
        ValueError: If the URL's domain isn't among approved API endpoints or 
                   if it attempts to point to a private network address.
    """
    parsed_url = urlparse(callback_url)
    
    # Check for internal IP addresses
    for internal_range in INTERNAL_IP_RANGES:
        if parsed_url.netloc.startswith(internal_range):
            raise ValueError(f"Access to internal network address is not allowed: {parsed_url.netloc}")
    
    # Check if domain is in approved list
    if parsed_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain not in approved API endpoints: {parsed_url.netloc}")
    
    # Log the validated URL
    print(f"Validated callback URL: {callback_url}")
    print(f"  Scheme: {parsed_url.scheme}")
    print(f"  Netloc: {parsed_url.netloc}")
    print(f"  Path: {parsed_url.path}")
    
    return callback_url
