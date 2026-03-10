from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def process_webhook(webhook_url: str) -> str:
    parsed = urlparse(webhook_url)
    
    # Extract the domain from the parsed URL
    domain = parsed.netloc.lower()
    
    # Check if domain is in the approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Check if the URL points to internal addresses
    for internal_range in INTERNAL_IP_RANGES:
        if domain.startswith(internal_range):
            raise ValueError(f"URL cannot point to internal network address: {domain}")
    
    # Return the validated URL
    return parsed.geturl()
