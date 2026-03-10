import urllib.parse
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}


def _validate_cdn_domain(base_cdn_url: str, allowed_domains: set) -> None:
    """
    Validates if the domain of the base_cdn_url is in the set of allowed domains.

    Args:
        base_cdn_url: The base URL of the CDN.
        allowed_domains: A set of allowed CDN domains.

    Raises:
        ValueError: If the domain is not an allowed CDN domain.
    """
    parsed_url = urllib.parse.urlparse(base_cdn_url)
    domain = parsed_url.netloc
    if domain not in allowed_domains:
        raise ValueError(
            f"Domain '{domain}' is not an allowed CDN domain. "
            f"Allowed domains are: {', '.join(allowed_domains)}"
        )


def _build_actual_url(base_cdn_url: str, res_type: str, res_id: str) -> str:
    """
    Constructs the full resource URL.

    Args:
        base_cdn_url: The base URL of the CDN.
        res_type: The type of the resource (e.g., 'image', 'script').
        res_id: The unique identifier of the resource.

    Returns:
        The complete URL for the resource.
    """
    # Ensure base_cdn_url ends with a slash for proper joining
    if not base_cdn_url.endswith('/'):
        base_cdn_url += '/'

    # Construct the path segment: res_type/res_id
    resource_path = f"{res_type}/{res_id}"

    # Join the base URL with the resource path
    full_url = urllib.parse.urljoin(base_cdn_url, resource_path)
    return full_url


def create_resource_url(base_cdn_url: str, res_id: str, res_type: str) -> str:
    """
    Builds a URL to retrieve certain types of resources hosted on external servers,
    using the base URL of a CDN and a user-defined resource identifier.

    Args:
        base_cdn_url: A string denoting the base URL of the CDN.
        res_id: A string specifying the unique identifier of the resource.
        res_type: A string indicating the kind of resource (e.g., 'image' or 'script').

    Returns:
        A string that forms the complete URL for the resource on the CDN.

    Raises:
        ValueError: If the constructed URL is not associated with any allowed CDN domains.
    """
    logger.info(
        f"Attempting to create resource URL with base_cdn_url='{base_cdn_url}', "
        f"res_id='{res_id}', res_type='{res_type}'"
    )
    _validate_cdn_domain(base_cdn_url, ALLOWED_CDN_DOMAINS)
    logger.debug("CDN domain validation successful.")

    full_url = _build_actual_url(base_cdn_url, res_type, res_id)
    logger.info(f"Successfully constructed URL: '{full_url}'")
    return full_url
