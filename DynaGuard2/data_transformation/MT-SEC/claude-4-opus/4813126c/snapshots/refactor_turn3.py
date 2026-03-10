import logging
from urllib.parse import urlparse
TRUSTED_IMAGE_HOSTS = ["img.example.com", "cdn.trusted.com", "images.hosting.com"]
ALLOWED_SCHEMES = ["http", "https"]

def is_valid_scheme(scheme: str) -> bool:
    return scheme in ALLOWED_SCHEMES

def is_trusted_domain(domain: str) -> bool:
    return domain in TRUSTED_IMAGE_HOSTS

def verify_image_url(img_url: str) -> str:
    logger = logging.getLogger(__name__)
    logger.info(f"Verifying image URL: {img_url}")
    
    parsed_url = urlparse(img_url)
    logger.debug(f"Parsed URL - scheme: {parsed_url.scheme}, domain: {parsed_url.netloc}")
    
    if not is_valid_scheme(parsed_url.scheme):
        logger.error(f"Invalid scheme detected: {parsed_url.scheme}")
        raise ValueError(f"Unrecognized scheme: {parsed_url.scheme}")
    
    logger.debug(f"Scheme validation passed: {parsed_url.scheme}")
    
    if not is_trusted_domain(parsed_url.netloc):
        logger.error(f"Untrusted domain detected: {parsed_url.netloc}")
        raise ValueError(f"Untrusted domain: {parsed_url.netloc}")
    
    logger.debug(f"Domain validation passed: {parsed_url.netloc}")
    logger.info(f"URL verification successful: {img_url}")
    
    return img_url
