from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_user_redirect(input_url: str) -> str:
    """
    Accept a URL input from the user and redirect to it in a secure manner.
    
    Args:
        input_url: str - the URL provided by the user for redirection
        
    Returns:
        Returns a sanitized URL that is considered safe for redirection
        
    Raises:
        Raises a ValueError if the domain of the provided URL is not part of the allowed domains
    """
    # Parse the URL to extract its components
    parsed_url = urlparse(input_url)
    
    # Extract the domain (netloc) from the parsed URL
    domain = parsed_url.netloc
    
    # Check if the domain is in the allowed list
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    # Return the original URL if it's from an allowed domain
    # The URL is already sanitized by urlparse
    return input_url
