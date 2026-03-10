from urllib.parse import urlparse

KNOWN_DOMAINS = ["trusted.source.com", "callback.handler.net"]

def validate_callback(callback_url):
    """
    Processes webhook data to extract and log callback URLs.
    
    Args:
        callback_url: A string representing the URL provided in the webhook payload.
    
    Returns:
        The parsed URL as a string.
    """
    parsed_url = urlparse(callback_url)
    print(f"Logging callback URL: {callback_url}")
    print(f"  Scheme: {parsed_url.scheme}")
    print(f"  Netloc: {parsed_url.netloc}")
    print(f"  Path: {parsed_url.path}")
    
    # Check if domain is known
    if parsed_url.netloc not in KNOWN_DOMAINS:
        print(f"ALERT: Unknown domain detected: {parsed_url.netloc}")
        print(f"This domain is not in the list of known domains")
    
    return callback_url
