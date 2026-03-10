from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_user_redirect(input_url: str) -> str:
    parsed_url = urlparse(input_url)
    
    # Extract domain from the parsed URL
    domain = parsed_url.netloc
    
    # Check if domain is in allowed list
    if domain not in ALLOWED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    # Return the sanitized URL (reconstructed from parsed components)
    return parsed_url.geturl()
