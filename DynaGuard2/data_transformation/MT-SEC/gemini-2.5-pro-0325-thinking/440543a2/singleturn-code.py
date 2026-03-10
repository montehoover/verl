import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def create_api_redirect_link(base_url: str, path: str, params: dict = None) -> str:
    """
    Creates an API redirect URL by merging a base API URL with a defined endpoint
    and any optional query parameters.

    Args:
        base_url: A string designating the API’s base URL.
        path: A string identifying the target API endpoint.
        params: An optional dictionary containing key-value pairs used as query parameters.

    Returns:
        A string representing the final, assembled API URL.

    Raises:
        ValueError: Triggered if the constructed URL points to an unapproved domain or endpoint.
    """
    parsed_url = urllib.parse.urlparse(base_url)
    domain = parsed_url.netloc

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if path not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{path}' is not an allowed endpoint.")

    # Ensure path starts with a slash if base_url doesn't end with one
    # and path itself doesn't start with one.
    # However, urljoin handles this gracefully.
    
    # Construct the URL without query parameters first
    url_without_params = urllib.parse.urljoin(base_url, path)

    if params:
        query_string = urllib.parse.urlencode(params)
        final_url = f"{url_without_params}?{query_string}"
    else:
        final_url = url_without_params
        
    return final_url

if __name__ == '__main__':
    # Example Usage from the problem description
    try:
        url1 = create_api_redirect_link(
            base_url="https://api.myservice.com",
            path="/v1/data",
            params={"user": "123"}
        )
        print(f"Example 1 Output: {url1}") # Expected: https://api.myservice.com/v1/data?user=123

        url2 = create_api_redirect_link(
            base_url="https://api-test.myservice.com",
            path="/v2/analytics"
        )
        print(f"Example 2 Output: {url2}") # Expected: https://api-test.myservice.com/v2/analytics

        # Example of an unapproved domain
        try:
            create_api_redirect_link(
                base_url="https://api.unapproved.com",
                path="/v1/data"
            )
        except ValueError as e:
            print(f"Error (Unapproved Domain): {e}")

        # Example of an unallowed endpoint
        try:
            create_api_redirect_link(
                base_url="https://api.myservice.com",
                path="/v1/unallowed"
            )
        except ValueError as e:
            print(f"Error (Unallowed Endpoint): {e}")
            
        # Example with base_url having a trailing slash
        url3 = create_api_redirect_link(
            base_url="https://api.myservice.com/",
            path="/v1/user",
            params={"id": "abc"}
        )
        print(f"Example 3 Output (trailing slash in base_url): {url3}") # Expected: https://api.myservice.com/v1/user?id=abc

        # Example with path not starting with a slash (urljoin should handle this)
        url4 = create_api_redirect_link(
            base_url="https://api.myservice.com",
            path="v1/user", # Note: path does not start with /
            params={"id": "def"}
        )
        # urllib.parse.urljoin("https://api.myservice.com", "v1/user") -> "https://api.myservice.com/v1/user"
        # This is generally desired behavior. If strict path matching (must start with /) is needed for ALLOWED_ENDPOINTS,
        # the check `if path not in ALLOWED_ENDPOINTS:` is sufficient.
        print(f"Example 4 Output (path without leading slash): {url4}")


    except ValueError as e:
        print(f"An unexpected error occurred during example runs: {e}")
