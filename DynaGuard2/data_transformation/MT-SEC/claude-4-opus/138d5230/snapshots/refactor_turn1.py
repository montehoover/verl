from urllib.parse import urlparse
ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def user_defined_redirect(redirect_url: str) -> str:
    """
    Accept a URL input from the user and redirect to it in a secure manner.
    
    Args:
        redirect_url: str, the URL provided by the user for redirection
        
    Returns:
        A sanitized URL that is considered safe for redirection
        
    Raises:
        ValueError: if the domain of the provided URL is not part of the allowed domains
    """
    # Parse the URL
    parsed_url = urlparse(redirect_url)
    
    # Extract the domain (netloc)
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of allowed redirect domains")
    
    # Return the sanitized URL (reconstructed from parsed components)
    return parsed_url.geturl()
