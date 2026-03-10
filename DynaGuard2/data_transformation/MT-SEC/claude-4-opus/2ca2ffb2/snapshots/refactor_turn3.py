import logging
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def validate_scheme(scheme: str) -> None:
    if scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unrecognized scheme: {scheme}")

def validate_domain(domain: str) -> None:
    if domain not in TRUSTED_IMAGE_HOSTS:
        raise ValueError(f"URL domain {domain} is not in the list of trusted domains")

def fetch_image_url(url: str) -> str:
    logger.info(f"Attempting to verify URL: {url}")
    
    try:
        parsed_url = urlparse(url)
        
        validate_scheme(parsed_url.scheme)
        validate_domain(parsed_url.netloc)
        
        logger.info(f"URL verification successful: {url}")
        return url
    except ValueError as e:
        logger.error(f"URL verification failed for {url}: {str(e)}")
        raise
