from urllib.parse import urlparse

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
    return callback_url
