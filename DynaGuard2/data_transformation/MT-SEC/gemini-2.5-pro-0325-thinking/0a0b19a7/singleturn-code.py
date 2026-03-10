import urllib.parse
from typing import Dict, Optional

# Setup code provided in the problem description
APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def build_api_redirect_url(api_base_url: str, endpoint: str, query_params: Optional[Dict[str, str]] = None) -> str:
    """
    Constructs a redirect URL for API responses, combining a base API URL
    with a user-provided endpoint and optional query parameters.

    Args:
        api_base_url: A string representing the base URL of the API
                      (e.g., "https://api.myservice.com").
        endpoint: A string representing the specific API endpoint
                  (e.g., "/v1/data"). Must be one of the paths defined
                  in ALLOWED_ENDPOINTS.
        query_params: An optional dictionary of query parameters (strings to strings).

    Returns:
        A string that is the complete API redirect URL.

    Raises:
        ValueError: If the api_base_url is malformed, or if the resulting URL's
                    domain is not in APPROVED_API_DOMAINS, or its path component
                    (derived from api_base_url and endpoint) is not in
                    ALLOWED_ENDPOINTS.
    """

    # Construct the URL part without query parameters using urljoin.
    # urljoin handles combining base URLs and relative/absolute paths correctly.
    # If endpoint starts with '/', it's treated as an absolute path relative to the domain of api_base_url.
    url_candidate = urllib.parse.urljoin(api_base_url, endpoint)

    # Parse the constructed candidate URL to validate its components.
    parsed_candidate_url = urllib.parse.urlparse(url_candidate)

    # Validate that the api_base_url and endpoint combination results in a URL with a scheme and domain.
    if not parsed_candidate_url.scheme or not parsed_candidate_url.netloc:
        raise ValueError(
            f"Invalid api_base_url ('{api_base_url}') or endpoint ('{endpoint}') combination. "
            f"Resulting candidate URL ('{url_candidate}') must have a scheme and domain."
        )

    # Validate the domain (netloc) of the candidate URL.
    if parsed_candidate_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_candidate_url.netloc}' from candidate URL '{url_candidate}' is not an approved API domain."
        )

    # Validate the path of the candidate URL.
    # Since all ALLOWED_ENDPOINTS start with '/', and urljoin with an absolute path for 'endpoint'
    # will make parsed_candidate_url.path equal to 'endpoint' (if api_base_url is well-formed).
    if parsed_candidate_url.path not in ALLOWED_ENDPOINTS:
        raise ValueError(
            f"Path '{parsed_candidate_url.path}' from candidate URL '{url_candidate}' is not an allowed endpoint."
        )

    # If all validations pass, build the final URL with query parameters.
    final_url = url_candidate
    if query_params:  # Checks if query_params is not None and not an empty dict
        query_string = urllib.parse.urlencode(query_params)
        if query_string:  # Ensure non-empty query string before appending '?'
            final_url += "?" + query_string
            
    return final_url
