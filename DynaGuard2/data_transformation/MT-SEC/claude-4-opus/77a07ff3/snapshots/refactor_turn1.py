from urllib.parse import urlparse

AUTHORIZED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def process_url_redirect(input_url: str) -> str:
    parsed_url = urlparse(input_url)
    
    if not parsed_url.scheme:
        raise ValueError("URL must include a scheme (http:// or https://)")
    
    if not parsed_url.netloc:
        raise ValueError("URL must include a domain")
    
    domain = parsed_url.netloc.lower()
    if domain.startswith('www.'):
        domain = domain[4:]
    
    if domain not in AUTHORIZED_REDIRECT_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not in the list of approved domains")
    
    return input_url
