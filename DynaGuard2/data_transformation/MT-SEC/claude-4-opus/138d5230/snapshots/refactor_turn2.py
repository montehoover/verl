from urllib.parse import urlparse

# Define the whitelist of trusted domains for secure redirection
ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]


def user_defined_redirect(redirect_url: str) -> str:
    """
    Accept a URL input from the user and redirect to it in a secure manner.
    
    This function validates that the provided URL's domain is within a predefined
    whitelist of allowed domains before returning the sanitized URL. This helps
    prevent open redirect vulnerabilities.
    
    Args:
        redirect_url (str): The URL provided by the user for redirection.
        
    Returns:
        str: A sanitized URL that is considered safe for redirection.
        
    Raises:
        ValueError: If the domain of the provided URL is not part of the 
                    allowed domains whitelist.
                    
    Example:
        >>> user_defined_redirect("https://trusted-redirect.com/page")
        'https://trusted-redirect.com/page'
        
        >>> user_defined_redirect("https://malicious.com/page")
        ValueError: Domain 'malicious.com' is not in the list of allowed redirect domains
    """
    # Parse the provided URL to extract its components
    parsed_url = urlparse(redirect_url)
    
    # Extract the domain (network location) from the parsed URL
    domain = parsed_url.netloc
    
    # Validate that the domain is in our whitelist of allowed domains
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(
            f"Domain '{domain}' is not in the list of allowed redirect domains"
        )
    
    # Return the sanitized URL by reconstructing it from parsed components
    # This ensures the URL is properly formatted and safe
    return parsed_url.geturl()
