from urllib.parse import urlparse
import requests

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def validate_webhook(webhook_link):
    # Parse the URL to check its structure
    parsed_url = urlparse(webhook_link)
    
    # Check if the URL has a valid scheme (http or https)
    if parsed_url.scheme not in ['http', 'https']:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only 'http' and 'https' are allowed.")
    
    # Check if the URL has a netloc (domain/host)
    if not parsed_url.netloc:
        raise ValueError("Invalid URL structure: missing domain/host.")
    
    # Extract the domain from the netloc (remove port if present)
    domain = parsed_url.netloc.split(':')[0]
    
    # Check if the domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain not in approved list: {domain}")
    
    # Check if the domain is trying to access internal addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"Access to internal addresses not allowed: {domain}")
    
    # Attempt to download the contents to ensure it's reachable
    try:
        response = requests.get(webhook_link, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        raise ValueError(f"URL is unreachable: {e}")
    
    return webhook_link
