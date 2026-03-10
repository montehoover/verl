from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_url_redirect(input_url: str) -> str:
    """
    Securely processes a URL for redirection by validating against authorized domains.
    
    Args:
        input_url: The URL input provided by the user
        
    Returns:
        A validated and safe-to-use URL for the redirection
        
    Raises:
        ValueError: If the domain of the provided URL is not in the list of approved domains
    """
    # Parse the URL to extract its components
    parsed_url = urlparse(input_url)
    
    # Extract the domain (netloc) from the parsed URL
    domain = parsed_url.netloc
    
    # Check if the domain is in the authorized list
    if domain not in AUTHORIZED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of authorized redirect domains")
    
    # Return the original URL if it passes validation
    return input_url
