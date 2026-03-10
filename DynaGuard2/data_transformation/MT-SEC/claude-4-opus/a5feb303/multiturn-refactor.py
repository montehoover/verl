from urllib.parse import urlparse
import logging
from datetime import datetime

ALLOWED_REDIRECT_DOMAINS = ["trusted-redirect.com", "partner.com"]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('redirect_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    logger.info(f"Redirection attempt - Input URL: {input_url}")
    
    domain = extract_domain(input_url)
    
    if not validate_domain(domain, ALLOWED_REDIRECT_DOMAINS):
        logger.warning(f"Redirection blocked - Domain '{domain}' not allowed - Input URL: {input_url}")
        raise ValueError(f"Domain '{domain}' is not in the allowed domains list")
    
    sanitized_url = sanitize_url(input_url)
    logger.info(f"Redirection allowed - Domain: {domain} - Sanitized URL: {sanitized_url}")
    
    return sanitized_url
