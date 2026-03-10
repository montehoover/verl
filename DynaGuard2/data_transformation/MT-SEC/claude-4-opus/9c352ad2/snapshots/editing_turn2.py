from urllib.parse import urlparse
import requests

def process_webhook(webhook_url: str) -> str:
    parsed = urlparse(webhook_url)
    
    # Check if the scheme is HTTPS
    if parsed.scheme != 'https':
        raise ValueError("Webhook URL must use HTTPS scheme")
    
    # Try to perform a HEAD request to check availability
    try:
        response = requests.head(webhook_url, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        raise ValueError("Webhook URL is not reachable")
    
    # Return the normalized URL
    return parsed.geturl()
