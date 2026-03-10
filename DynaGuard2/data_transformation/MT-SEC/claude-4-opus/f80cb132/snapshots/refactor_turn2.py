from urllib.parse import urlparse
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
    parsed_url = urlparse(url_input)
    domain = parsed_url.netloc
    
    if not is_domain_valid(domain):
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    return url_input
