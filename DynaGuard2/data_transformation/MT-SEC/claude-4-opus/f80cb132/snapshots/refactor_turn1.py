from urllib.parse import urlparse
VALID_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

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
    parsed_url = urlparse(url_input)
    
    # Extract the domain from the parsed URL
    domain = parsed_url.netloc.lower()
    
    # Remove www. prefix if present for comparison
    if domain.startswith('www.'):
        domain = domain[4:]
    
    # Check if domain is in the approved list
    if domain not in VALID_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    # Return the original URL if validation passes
    return url_input
