import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_redirect_url(main_url: str, endpoint_name: str, params_dict: dict = None) -> str:
    """
    Creates a full redirect URL for API requests by combining the base API URL,
    a specific endpoint, and optional query parameters.

    Args:
        main_url: String representing the main API base URL.
        endpoint_name: String representing the desired API endpoint.
        params_dict: An optional dictionary for query string parameters.

    Returns:
        Produces a complete API redirect URL as a string.

    Raises:
        ValueError: If the generated URL diverges from an approved domain or endpoint.
    """
    parsed_main_url = urllib.parse.urlparse(main_url)
    
    # Validate domain
    if parsed_main_url.hostname not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{parsed_main_url.hostname}' is not an approved API domain.")

    # Validate endpoint
    if endpoint_name not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{endpoint_name}' is not an allowed endpoint.")

    # Ensure main_url doesn't end with a slash if endpoint_name starts with one
    if main_url.endswith('/') and endpoint_name.startswith('/'):
        url_parts = [main_url.rstrip('/'), endpoint_name]
    elif not main_url.endswith('/') and not endpoint_name.startswith('/'):
        url_parts = [main_url, '/', endpoint_name]
    else:
        url_parts = [main_url, endpoint_name]
    
    full_url_path = "".join(url_parts)

    if params_dict:
        query_string = urllib.parse.urlencode(params_dict)
        return f"{full_url_path}?{query_string}"
    else:
        return full_url_path

if __name__ == '__main__':
    # Example Usage (matches the example input/output)
    try:
        url1 = construct_redirect_url(
            main_url="https://api.myservice.com",
            endpoint_name="/v1/data",
            params_dict={"user": "123"}
        )
        print(f"Example 1: {url1}") # Expected: https://api.myservice.com/v1/data?user=123

        url2 = construct_redirect_url(
            main_url="https://api-test.myservice.com",
            endpoint_name="/v2/analytics"
        )
        print(f"Example 2: {url2}") # Expected: https://api-test.myservice.com/v2/analytics

        # Example of an invalid domain
        try:
            construct_redirect_url(
                main_url="https://api.unapproved.com",
                endpoint_name="/v1/data"
            )
        except ValueError as e:
            print(f"Error (Invalid Domain): {e}")

        # Example of an invalid endpoint
        try:
            construct_redirect_url(
                main_url="https://api.myservice.com",
                endpoint_name="/v3/unknown"
            )
        except ValueError as e:
            print(f"Error (Invalid Endpoint): {e}")
            
        # Example with main_url ending with / and endpoint starting with /
        url3 = construct_redirect_url(
            main_url="https://api.myservice.com/",
            endpoint_name="/v1/user",
            params_dict={"id": "456"}
        )
        print(f"Example 3 (slash handling): {url3}") # Expected: https://api.myservice.com/v1/user?id=456

        # Example with main_url not ending with / and endpoint not starting with /
        # This case should ideally be avoided by ensuring endpoint_name always starts with /
        # but the code handles it by adding a /
        url4 = construct_redirect_url(
            main_url="https://api.myservice.com",
            endpoint_name="health" # Assuming 'health' should be '/health'
        )
        # To make this work as expected, 'health' should be in ALLOWED_ENDPOINTS or '/health' should be passed.
        # For now, let's test with '/health' as per ALLOWED_ENDPOINTS
        url4_corrected = construct_redirect_url(
            main_url="https://api.myservice.com",
            endpoint_name="/health" 
        )
        print(f"Example 4 (corrected endpoint): {url4_corrected}") # Expected: https://api.myservice.com/health


    except ValueError as e:
        print(f"An unexpected error occurred during example runs: {e}")
