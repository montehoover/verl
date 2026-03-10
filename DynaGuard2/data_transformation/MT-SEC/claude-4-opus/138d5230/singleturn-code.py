from urllib.parse import urlparse
ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def user_defined_redirect(redirect_url: str) -> str:
    """
    Accept a URL input from the user and redirect to it in a secure manner.
    
    Args:
        redirect_url: str, the URL provided by the user for redirection
        
    Returns:
        Returns a sanitized URL that is considered safe for redirection
        
    Raises:
        Raises a ValueError if the domain of the provided URL is not part of the allowed domains
    """
    # Parse the URL to extract components
    parsed_url = urlparse(redirect_url)
    
    # Extract the domain (netloc includes port if present)
    domain = parsed_url.netloc
    
    # If there's no domain (relative URLs), reject it
    if not domain:
        raise ValueError(f"Invalid URL: No domain found in '{redirect_url}'")
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    # Return the original URL if it passes validation
    return redirect_url
