import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def construct_resource_url(base_cdn_url, res_id, res_type):
    """
    Constructs a resource URL from base CDN URL, resource ID, and resource type.
    
    Args:
        base_cdn_url (str): The base URL of the CDN.
        res_id (str): The unique identifier of the resource.
        res_type (str): The type of resource (e.g., 'image', 'script').
    
    Returns:
        str: The complete URL path to the resource.
    """
    return urllib.parse.urljoin(base_cdn_url, f"{res_type}/{res_id}")


def validate_cdn_domain(url, allowed_domains):
    """
    Validates that the URL's domain is in the allowed CDN domains set.
    
    Args:
        url (str): The URL to validate.
        allowed_domains (set): Set of allowed CDN domain names.
    
    Returns:
        bool: True if the domain is valid.
    
    Raises:
        ValueError: If the URL domain is not in the allowed domains.
    """
    parsed_url = urllib.parse.urlparse(url)
    domain = parsed_url.netloc
    
    if domain not in allowed_domains:
        raise ValueError(f"URL domain '{domain}' is not in the allowed CDN domains")
    
    return True


def create_resource_url(base_cdn_url, res_id, res_type):
    """
    Builds a URL to retrieve resources hosted on external CDN servers.
    
    This function constructs a complete URL for accessing resources like images
    or scripts from a CDN, validating that the CDN domain is allowed.
    
    Args:
        base_cdn_url (str): The base URL of the CDN.
        res_id (str): The unique identifier of the resource.
        res_type (str): The type of resource (e.g., 'image', 'script').
    
    Returns:
        str: The complete URL for the resource on the CDN.
    
    Raises:
        ValueError: If the constructed URL is not associated with any allowed
                    CDN domains.
    """
    logger.info(f"Building resource URL - Base: {base_cdn_url}, Type: {res_type}, ID: {res_id}")
    
    # Construct the complete URL
    url = construct_resource_url(base_cdn_url, res_id, res_type)
    logger.debug(f"Constructed URL: {url}")
    
    # Validate the domain
    validate_cdn_domain(url, ALLOWED_CDN_DOMAINS)
    logger.info(f"Successfully validated and constructed URL: {url}")
    
    return url
