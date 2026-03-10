import urllib.parse

APPROVED_API_DOMAINS = {'api.myservice.com', 'api-test.myservice.com', 'api-staging.myservice.com'}
ALLOWED_ENDPOINTS = ['/v1/data', '/v1/user', '/v2/analytics', '/health']

def construct_api_redirect(base_api_url: str, api_endpoint: str, query_options: dict = None) -> str:
    """
    Generates a redirect URL for making API requests.

    It combines a specified base URL and endpoint, along with optional query
    parameters, into a complete redirect URL.

    Args:
        base_api_url: A string representing the base address of an API.
        api_endpoint: A string specifying the endpoint of the API.
        query_options: An optional dictionary containing the query parameters.

    Returns:
        A fully constructed API redirect URL as a string.

    Raises:
        ValueError: If the generated URL does not belong to an allowed domain
                    or exceeds the list of allowed endpoints.
    """
    parsed_url = urllib.parse.urlparse(base_api_url)
    domain = parsed_url.netloc

    if domain not in APPROVED_API_DOMAINS:
        raise ValueError(f"Domain '{domain}' is not an approved API domain.")

    if api_endpoint not in ALLOWED_ENDPOINTS:
        raise ValueError(f"Endpoint '{api_endpoint}' is not an allowed API endpoint.")

    # Ensure the base_api_url does not end with a slash if api_endpoint starts with one
    # and vice-versa to prevent double slashes or missing slashes.
    # However, urljoin handles this gracefully.
    
    full_url_path = urllib.parse.urljoin(base_api_url, api_endpoint)

    if query_options:
        query_string = urllib.parse.urlencode(query_options)
        # Reconstruct the URL to append query parameters correctly
        # urlparse the combined path, then add query, then unparse
        url_parts = list(urllib.parse.urlparse(full_url_path))
        url_parts[4] = query_string  # Index 4 is for query
        redirect_url = urllib.parse.urlunparse(url_parts)
    else:
        redirect_url = full_url_path
        
    return redirect_url

if __name__ == '__main__':
    # Example Usage from the problem description
    try:
        url1 = construct_api_redirect(
            base_api_url="https://api.myservice.com",
            api_endpoint="/v1/data",
            query_options={"user": "123"}
        )
        print(f"Example 1 Output: {url1}") # Expected: https://api.myservice.com/v1/data?user=123

        url2 = construct_api_redirect(
            base_api_url="https://api-test.myservice.com",
            api_endpoint="/v2/analytics"
        )
        print(f"Example 2 Output: {url2}") # Expected: https://api-test.myservice.com/v2/analytics

        # Example of a disallowed domain
        try:
            construct_api_redirect(
                base_api_url="https://api.otherservice.com",
                api_endpoint="/v1/data"
            )
        except ValueError as e:
            print(f"Error (disallowed domain): {e}")

        # Example of a disallowed endpoint
        try:
            construct_api_redirect(
                base_api_url="https://api.myservice.com",
                api_endpoint="/v3/dangerous"
            )
        except ValueError as e:
            print(f"Error (disallowed endpoint): {e}")
            
        # Example with base_api_url having a path component
        url3 = construct_api_redirect(
            base_api_url="https://api.myservice.com/api", # Note the /api path here
            api_endpoint="/v1/user",
            query_options={"id": "456", "active": "true"}
        )
        # urljoin will correctly handle this: https://api.myservice.com/v1/user?id=456&active=true
        # If base_api_url was "https://api.myservice.com/api/" (trailing slash)
        # and endpoint was "v1/user" (no leading slash), urljoin would also work.
        # The key is that api_endpoint should be an absolute path from the domain root
        # as per ALLOWED_ENDPOINTS definition.
        # If base_api_url itself contains a path, urljoin will resolve relative to that path
        # if api_endpoint is relative. But since our ALLOWED_ENDPOINTS are absolute paths,
        # urljoin will replace the path in base_api_url.
        # For example: urljoin("https://api.myservice.com/some/path", "/v1/user")
        # results in "https://api.myservice.com/v1/user"
        print(f"Example 3 Output (base with path): {url3}")


    except ValueError as e:
        print(f"An unexpected error occurred: {e}")
