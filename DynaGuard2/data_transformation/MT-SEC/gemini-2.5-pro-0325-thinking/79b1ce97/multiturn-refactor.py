import urllib.parse
import logging

# Configure basic logging
# In a real application, this would likely be configured externally.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_url_components(base_url_api: str, desired_endpoint: str, approved_domains: set, allowed_endpoints: list) -> None:
    """
    Validates the domain of the base URL and the desired endpoint.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        approved_domains: A set of approved API domains.
        allowed_endpoints: A list of allowed API endpoints.

    Raises:
        ValueError: If the domain or endpoint is not approved/allowed.
    """
    logger.info(
        f"Validating URL components: base_url_api='{base_url_api}', desired_endpoint='{desired_endpoint}'"
    )
    parsed_base_url = urllib.parse.urlparse(base_url_api)
    
    if parsed_base_url.hostname not in approved_domains:
        logger.error(
            f"Validation failed: Domain '{parsed_base_url.hostname}' is not in approved domains {approved_domains}."
        )
        raise ValueError(f"Domain '{parsed_base_url.hostname}' is not an approved API domain.")
    logger.debug(f"Domain '{parsed_base_url.hostname}' is approved.")

    if desired_endpoint not in allowed_endpoints:
        logger.error(
            f"Validation failed: Endpoint '{desired_endpoint}' is not in allowed endpoints {allowed_endpoints}."
        )
        raise ValueError(f"Endpoint '{desired_endpoint}' is not an allowed endpoint.")
    logger.debug(f"Endpoint '{desired_endpoint}' is allowed.")
    logger.info("URL components validated successfully.")


def _construct_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Constructs the full URL from base URL, endpoint, and optional parameters.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        optional_params: An optional dictionary for query string parameters.

    Returns:
        The constructed full URL as a string.
    """
    logger.info(
        f"Constructing URL: base_url_api='{base_url_api}', desired_endpoint='{desired_endpoint}', optional_params={optional_params}"
    )
    # urllib.parse.urljoin is robust for joining base URLs and paths.
    # If desired_endpoint is an absolute path (starts with '/'),
    # it replaces the path component of base_url_api.
    # e.g., urljoin("https://api.example.com/initial/path", "/v1/new")
    # results in "https://api.example.com/v1/new"
    url_without_query = urllib.parse.urljoin(base_url_api, desired_endpoint)
    logger.debug(f"URL without query parameters: '{url_without_query}'")

    if optional_params:
        query_string = urllib.parse.urlencode(optional_params)
        final_url = f"{url_without_query}?{query_string}"
        logger.debug(f"Encoded query string: '{query_string}'")
    else:
        final_url = url_without_query
    
    logger.info(f"Constructed URL: '{final_url}'")
    return final_url


def create_redirect_url(base_url_api: str, desired_endpoint: str, optional_params: dict = None) -> str:
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        base_url_api: String representing the main API base URL.
        desired_endpoint: String representing the desired API endpoint.
        optional_params: An optional dictionary for query string parameters.

    Returns:
        Produces a complete API redirect URL as a string.

    Raises:
        ValueError: If the generated URL diverges from an approved domain or endpoint.
    """
    logger.info(
        f"Creating redirect URL with base_url_api='{base_url_api}', "
        f"desired_endpoint='{desired_endpoint}', optional_params={optional_params}"
    )
    
    try:
        _validate_url_components(base_url_api, desired_endpoint, APPROVED_API_DOMAINS, ALLOWED_ENDPOINTS)
        logger.debug("URL components validation successful in create_redirect_url context.")
    except ValueError as e:
        logger.error(f"Error during URL validation: {e}")
        raise  # Re-raise the caught exception

    final_url = _construct_url(base_url_api, desired_endpoint, optional_params)
    logger.info(f"Successfully created redirect URL: '{final_url}'")
    
    return final_url
