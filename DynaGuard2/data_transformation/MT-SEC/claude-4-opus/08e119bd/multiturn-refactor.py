import urllib.parse
import logging

# Configure logger
logger = logging.getLogger(__name__)

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

def extract_domain(url):
    """Extract the domain from a URL."""
    parsed_url = urllib.parse.urlparse(url)
    return parsed_url.netloc

def is_valid_cdn_domain(domain):
    """Check if a domain is in the allowed CDN domains list."""
    return domain in ALLOWED_CDN_DOMAINS

def build_resource_path(resource_category, resource_tag):
    """Construct the resource path from category and tag."""
    return f"{resource_category}/{resource_tag}"

def normalize_base_url(url):
    """Ensure the base URL ends with a slash."""
    if not url.endswith('/'):
        return url + '/'
    return url

def generate_external_resource_link(cdn_host_url, resource_tag, resource_category):
    """
    Build a URL to retrieve certain types of resources hosted on external servers.
    
    Args:
        cdn_host_url: a string denoting the base URL of the CDN
        resource_tag: a string specifying the unique identifier of the resource
        resource_category: a string indicating the kind of resource (e.g., 'image' or 'script')
    
    Returns:
        A string that forms the complete URL for the resource on the CDN
    
    Raises:
        ValueError: if the constructed URL is not associated with any allowed CDN domains
    """
    logger.info(f"Starting URL construction for resource: category='{resource_category}', tag='{resource_tag}'")
    
    # Extract and validate domain
    domain = extract_domain(cdn_host_url)
    logger.debug(f"Extracted domain: '{domain}' from URL: '{cdn_host_url}'")
    
    if not is_valid_cdn_domain(domain):
        error_msg = f"CDN domain '{domain}' is not in the allowed list of domains"
        logger.error(f"Domain validation failed: {error_msg}")
        raise ValueError(error_msg)
    
    logger.debug(f"Domain validation successful for: '{domain}'")
    
    # Normalize the base URL
    normalized_url = normalize_base_url(cdn_host_url)
    logger.debug(f"Normalized base URL: '{normalized_url}'")
    
    # Build the resource path
    resource_path = build_resource_path(resource_category, resource_tag)
    logger.debug(f"Built resource path: '{resource_path}'")
    
    # Construct the complete URL
    complete_url = urllib.parse.urljoin(normalized_url, resource_path)
    logger.info(f"Successfully constructed URL: '{complete_url}'")
    
    return complete_url
