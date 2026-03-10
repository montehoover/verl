import re
import logging
from urllib.parse import urlparse

ALLOWED_API_DOMAINS = ["api.trustedsource.com", "gateway.partnerapi.com"]
INTERNAL_IP_PATTERNS = [
    r"^localhost$",                # Localhost
    r"^127\.0\.0\.\d+$",           # Loopback address
    r"^10\.\d{1,3}\.\d{1,3}\.\d{1,3}$",  # Private 10.x.x.x range
    r"^192\.168\.\d{1,3}\.\d{1,3}$"      # Private 192.168.x.x range
]

def _parse_url(endpoint_address: str) -> tuple:
    """Parse URL and return parsed URL object and hostname."""
    parsed_url = urlparse(endpoint_address)
    hostname = parsed_url.hostname
    if not hostname:
        raise ValueError("Invalid URL: no hostname found")
    return parsed_url, hostname

def _validate_scheme(parsed_url) -> None:
    """Ensure the URL uses HTTPS protocol."""
    if parsed_url.scheme != 'https':
        raise ValueError("Only HTTPS URLs are allowed for secure API access")

def _is_internal_ip(hostname: str) -> bool:
    """Check if hostname matches any internal IP pattern."""
    for pattern in INTERNAL_IP_PATTERNS:
        if re.match(pattern, hostname):
            return True
    return False

def _validate_no_internal_ip(hostname: str) -> None:
    """Ensure hostname is not an internal IP address."""
    if _is_internal_ip(hostname):
        raise ValueError(f"Access to internal network address '{hostname}' is not allowed")

def _validate_allowed_domain(hostname: str) -> None:
    """Ensure hostname is in the allowed domains list."""
    if hostname not in ALLOWED_API_DOMAINS:
        raise ValueError(f"Domain '{hostname}' is not in the list of allowed API domains")

def verify_api_address(endpoint_address: str) -> str:
    # Initialize logger
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting API address verification for URL: {endpoint_address}")
    
    try:
        # Pipeline of validation steps
        logger.debug("Parsing URL and extracting hostname")
        parsed_url, hostname = _parse_url(endpoint_address)
        logger.debug(f"Successfully parsed URL. Hostname: {hostname}, Scheme: {parsed_url.scheme}")
        
        logger.debug("Validating URL scheme")
        _validate_scheme(parsed_url)
        logger.debug("URL scheme validation passed - using HTTPS")
        
        logger.debug(f"Checking if hostname '{hostname}' is an internal IP address")
        _validate_no_internal_ip(hostname)
        logger.debug("Internal IP validation passed - not an internal address")
        
        logger.debug(f"Validating if domain '{hostname}' is in allowed list")
        _validate_allowed_domain(hostname)
        logger.debug(f"Domain validation passed - '{hostname}' is allowed")
        
        logger.info(f"API address verification successful for URL: {endpoint_address}")
        # Return the verified URL
        return endpoint_address
        
    except ValueError as e:
        logger.error(f"API address verification failed for URL: {endpoint_address}. Error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during API address verification for URL: {endpoint_address}. Error: {str(e)}")
        raise
