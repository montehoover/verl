from urllib.parse import urlparse
import requests
import re

VALID_SCHEMES = ["http", "https"]
API_PATTERN = r'.*\.api\..*'

def validate_webhook(webhook_link):
    # Parse the URL to check its structure
    parsed_url = urlparse(webhook_link)
    
    # Check if the URL has a valid scheme (http or https)
    if parsed_url.scheme not in VALID_SCHEMES:
        raise ValueError(f"Invalid URL scheme: {parsed_url.scheme}. Only 'http' and 'https' are allowed.")
    
    # Check if the URL has a netloc (domain/host)
    if not parsed_url.netloc:
        raise ValueError("Invalid URL structure: missing domain/host.")
    
    # Check if the domain resembles an API endpoint structure
    if not re.match(API_PATTERN, parsed_url.netloc):
        raise ValueError(f"Domain does not match API pattern: {parsed_url.netloc}")
    
    # Attempt to download the contents to ensure it's reachable
    try:
        response = requests.get(webhook_link, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
    except requests.exceptions.RequestException as e:
        raise ValueError(f"URL is unreachable: {e}")
    
    return webhook_link
