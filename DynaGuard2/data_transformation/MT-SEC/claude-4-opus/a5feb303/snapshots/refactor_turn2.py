from urllib.parse import urlparse

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

def extract_domain(url: str) -> str:
    """Extract the domain from a URL."""
    parsed_url = urlparse(url)
    return parsed_url.netloc

def validate_domain(domain: str, allowed_domains: list[str]) -> bool:
    """Check if a domain is in the allowed domains list."""
    return domain in allowed_domains

def sanitize_url(url: str) -> str:
    """Sanitize and reconstruct the URL."""
    parsed_url = urlparse(url)
    return parsed_url.geturl()

def process_user_redirect(input_url: str) -> str:
    domain = extract_domain(input_url)
    
    if not validate_domain(domain, ALLOWED_REDIRECT_DOMAINS):
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    return sanitize_url(input_url)
