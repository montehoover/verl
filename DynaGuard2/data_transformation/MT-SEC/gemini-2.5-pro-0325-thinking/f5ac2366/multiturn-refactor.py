import urllib.parse
import logging

# Basic logging configuration (configure as needed for your application)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _validate_cdn_domain(cdn_base_url: str, allowed_domains: set) -> None:
    """
    Validates if the domain of the cdn_base_url is in the set of allowed domains.

    Args:
        cdn_base_url: The base URL of the CDN.
        allowed_domains: A set of allowed CDN domain strings.

    Raises:
        ValueError: If the domain of cdn_base_url is not in allowed_domains.
    """
    parsed_base_url = urllib.parse.urlparse(cdn_base_url)
    if parsed_base_url.netloc not in allowed_domains:
        raise ValueError(
            f"Domain '{parsed_base_url.netloc}' is not an allowed CDN domain. "
            f"Allowed domains are: {', '.join(allowed_domains)}"
        )


def _build_url_path(base_url: str, resource_type: str, resource_id: str) -> str:
    """
    Constructs the full URL path for a resource.

    Args:
        base_url: The base URL (e.g., CDN base).
        resource_type: The type of resource (e.g., 'image', 'script').
        resource_id: The identifier of the resource.

    Returns:
        The complete URL string.
    """
    # Ensure base_url ends with a slash for proper joining
    if not base_url.endswith('/'):
        base_url += '/'

    # Using urllib.parse.urljoin for robust path construction
    # First join base with resource_type
    path_part_one = urllib.parse.urljoin(base_url, resource_type)
    # Ensure the result of the first join has a trailing slash before joining the resource_id
    if not path_part_one.endswith('/'):
        path_part_one += '/'
    
    full_url = urllib.parse.urljoin(path_part_one, resource_id)
    return full_url


def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Constructs a URL for accessing external resources based on a user-provided
    resource identifier and a base CDN URL.

    Args:
        cdn_base_url: The base URL of the CDN.
        resource_id: The identifier of the resource to be fetched.
        resource_type: The type of resource (e.g., 'image', 'script').

    Returns:
        A string that is the complete URL for accessing the external resource.

    Raises:
        ValueError: If the resulting URL is not pointing to an allowed CDN domain.
    """
    logger.info(
        f"Attempting to construct URL with base: '{cdn_base_url}', "
        f"type: '{resource_type}', id: '{resource_id}'"
    )
    try:
        _validate_cdn_domain(cdn_base_url, ALLOWED_CDN_DOMAINS)
        
        full_url = _build_url_path(cdn_base_url, resource_type, resource_id)
        
        logger.info(
            f"Successfully constructed URL: '{full_url}' for base: '{cdn_base_url}', "
            f"type: '{resource_type}', id: '{resource_id}'"
        )
        return full_url
    except ValueError as e:
        logger.error(
            f"Failed to construct URL for base: '{cdn_base_url}', type: '{resource_type}', id: '{resource_id}'. "
            f"Error: {e}"
        )
        raise
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during URL construction for base: '{cdn_base_url}', "
            f"type: '{resource_type}', id: '{resource_id}'. Error: {e}",
            exc_info=True # Adds stack trace for unexpected errors
        )
        raise
