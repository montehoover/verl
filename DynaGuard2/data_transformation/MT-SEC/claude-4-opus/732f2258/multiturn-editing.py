from urllib.parse import urlparse

APPROVED_API_DOMAINS = ["api.service.com", "hooks.webapp.com"]
INTERNAL_IP_RANGES = ["localhost", "127.0.0.1", "10.", "192.168."]

def extract_url_components(url):
    parsed = urlparse(url)
    
    return {
        'scheme': parsed.scheme,
        'domain': parsed.netloc,
        'path': parsed.path,
        'params': parsed.params,
        'query': parsed.query,
        'fragment': parsed.fragment,
        'username': parsed.username,
        'password': parsed.password,
        'hostname': parsed.hostname,
        'port': parsed.port
    }

def check_domain_approval(url, allowed_domains):
    parsed = urlparse(url)
    domain = parsed.netloc
    
    is_approved = domain in allowed_domains
    message = 'approved' if is_approved else 'not approved'
    
    return {
        'domain': domain,
        'message': message,
        'is_approved': is_approved
    }

def analyze_webhook(webhook_address):
    parsed = urlparse(webhook_address)
    domain = parsed.netloc
    hostname = parsed.hostname or domain
    
    # Check if domain is in approved list
    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the approved API domains list")
    
    # Check if pointing to internal network
    for internal_range in INTERNAL_IP_RANGES:
        if hostname.startswith(internal_range):
            raise ValueError(f"URL points to internal network: {hostname}")
    
    return webhook_address
