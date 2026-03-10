from urllib.parse import urlparse
VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def secure_user_redirect(url_input: str) -> str:
    """
    Securely process a user-provided URL for redirection.
    
    Args:
        url_input: str - The URL input provided by the user
        
    Returns:
        str - A validated and safe-to-use URL for the redirection
        
    Raises:
        ValueError - If the domain of the provided URL is not in the list of approved domains
    """
    # Parse the URL to extract components
    parsed_url = urlparse(url_input)
    
    # Extract the domain (netloc contains domain:port if port is specified)
    domain = parsed_url.netloc
    
    # Handle cases where domain might include port number
    if ':' in domain:
        domain = domain.split(':')[0]
    
    # Check if the domain is in the approved list
    if domain not in VALID_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    # Return the original URL if validation passes
    return url_input
