"""
Module for generating secure API redirect URLs.

This module provides functionality to construct API redirect URLs while ensuring
they adhere to a predefined set of approved domains and allowed endpoints.
It uses helper functions to validate URL components and construct the final URL,
promoting modularity and security.
"""
import urllib.parse

# --- Constants ---

# Set of approved API domains to prevent redirection to malicious sites.
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}

# List of allowed API endpoints to restrict access to defined paths.
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


# --- Helper Functions ---

def _validate_url_components(base_url: str, api_path: str):
    """
    Validates the base URL domain and API path against approved lists.

    Args:
        base_url: The base URL string.
        api_path: The API path string.

    Raises:
        ValueError: If the domain is not in APPROVED_API_DOMAINS or
                    the api_path is not in ALLOWED_ENDPOINTS.
    """
    # Parse the base_url to extract its components, especially the hostname.
    parsed_base_url = urllib.parse.urlparse(base_url)

    # Validate that the hostname of the base_url is in the set of approved domains.
    if parsed_base_url.hostname not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_base_url.hostname}' is not an approved API domain."
        )

    # Validate that the api_path is in the list of allowed endpoints.
    if api_path not in ALLOWED_ENDPOINTS:
        raise ValueError(
            f"Endpoint '{api_path}' is not an allowed API endpoint."
        )


def _construct_url(base_url: str, api_path: str, params: dict = None) -> str:
    """
    Constructs the full API URL from base URL, API path, and optional parameters.

    Args:
        base_url: The base URL string.
        api_path: The API path string.
        params: An optional dictionary of query parameters.

    Returns:
        The fully constructed URL string.
    """
    # Ensure the base_url ends with a slash for proper joining with api_path.
    # Also, ensure api_path does not start with a slash if base_url already has one,
    # though urljoin typically handles this gracefully.
    # Using urljoin is preferred as it correctly handles various base URL and path combinations.
    normalized_base_url = base_url if base_url.endswith('/') else base_url + '/'
    normalized_api_path = api_path.lstrip('/')
    
    full_path = urllib.parse.urljoin(normalized_base_url, normalized_api_path)

    # If query parameters are provided, encode them and append to the URL.
    if params:
        query_string = urllib.parse.urlencode(params)
        return f"{full_path}?{query_string}"
    else:
        # If no parameters, return the URL as is.
        return full_path


# --- Main Function ---

def generate_api_redirect(base_url: str, api_path: str, params: dict = None) -> str:
    """
    Generates a redirect URL for making API requests.

    Combines a specified base URL and endpoint, along with optional query
    parameters, into a complete redirect URL.

    Args:
        base_url: A string representing the base address of an API.
        api_path: A string specifying the endpoint of the API.
        params: An optional dictionary containing the query parameters.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
                    or uses a disallowed endpoint.
    """
    # First, validate the provided base_url and api_path.
    # This will raise a ValueError if any component is invalid.
    _validate_url_components(base_url, api_path)

    # If validation passes, construct the full URL.
    return _construct_url(base_url, api_path, params)
