import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']


def _validate_url_components(main_url: str, endpoint_name: str) -> None:
    """
    Validates the domain of the main_url and the endpoint_name.

    Args:
        main_url: The base API URL.
        endpoint_name: The API endpoint.

    Raises:
        ValueError: If the domain or endpoint is not approved/allowed.
    """
    parsed_main_url = urllib.parse.urlparse(main_url)
    if parsed_main_url.netloc not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_main_url.netloc}' is not an approved API domain.")

    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not an allowed endpoint.")


def _build_url_with_params(main_url: str, endpoint_name: str, params_dict: dict = None) -> str:
    """
    Constructs the full URL by joining the main_url with the endpoint_name
    and appending query parameters if provided.

    Args:
        main_url: The base API URL (assumed to be validated).
        endpoint_name: The API endpoint (assumed to be validated).
        params_dict: An optional dictionary for query string parameters.

    Returns:
        The complete API redirect URL as a string.
    """
    # urljoin correctly handles joining base URLs and paths,
    # e.g., 'http://example.com/foo' + '/bar' -> 'http://example.com/bar'
    # and 'http://example.com/foo/' + 'bar' -> 'http://example.com/foo/bar'
    base_redirect_url = urllib.parse.urljoin(main_url, endpoint_name)

    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        # Use urlparse and urlunparse to correctly add the query string
        # to the base_redirect_url, preserving its existing components.
        url_parts = list(urllib.parse.urlparse(base_redirect_url))
        url_parts[4] = query_string  # Index 4 is for the query component
        redirect_url = urllib.parse.urlunparse(url_parts)
    else:
        redirect_url = base_redirect_url
    
    # Post-construction validation of the domain can be useful as a safeguard,
    # particularly if urljoin had unexpected behavior with unusual inputs,
    # though _validate_url_components should catch most issues.
    parsed_redirect_url = urllib.parse.urlparse(redirect_url)
    if parsed_redirect_url.netloc not in APPROVED_API_DOMAINS:
        # This might indicate an issue with how main_url was formed or joined.
        raise ValueError(f"Constructed URL domain '{parsed_redirect_url.netloc}' is not approved. "
                         "This may indicate an issue with the base URL or joining logic.")

    return redirect_url


def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: dict = None) -> str:
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        main_url: string representing the main API base URL.
        endpoint_name: string representing the desired API endpoint.
        params_dict: an optional dictionary for query string parameters.

    Returns:
        Produces a complete API redirect URL as a string.

    Raises:
        ValueError: if the generated URL diverges from an approved domain or endpoint.
    """
    _validate_url_components(main_url, endpoint_name)
    redirect_url = _build_url_with_params(main_url, endpoint_name, params_dict)
    return redirect_url
