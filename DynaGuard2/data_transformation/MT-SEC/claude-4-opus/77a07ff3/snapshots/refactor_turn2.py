from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_domain(url: str) -> str:
    """Extract and normalize the domain from a URL."""
    parsed_url = urlparse(url)
    
    if not parsed_url.scheme:
        raise ValueError("URL must include a scheme (http:// or https://)")
    
    if not parsed_url.netloc:
        raise ValueError("URL must include a domain")
    
    domain = parsed_url.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    
    return domain

def is_domain_authorized(domain: str) -> bool:
    """Check if a domain is in the list of authorized redirect domains."""
    return domain in AUTHORIZED_REDIRECT_DOMAINS

def process_url_redirect(input_url: str) -> str:
    domain = extract_domain(input_url)
    
    if not is_domain_authorized(domain):
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    return input_url
