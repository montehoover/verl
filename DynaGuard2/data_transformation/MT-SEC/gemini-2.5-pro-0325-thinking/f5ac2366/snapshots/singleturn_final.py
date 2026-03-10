import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}

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
    # Ensure cdn_base_url does not end with a slash to avoid double slashes
    if cdn_base_url.endswith('/'):
        cdn_base_url = cdn_base_url[:-1]

    # Construct the path part of the URL
    # Ensure resource_type and resource_id are properly escaped if they can contain special characters
    # For this example, assuming they are simple strings.
    path_part = f"/{urllib.parse.quote(resource_type)}/{urllib.parse.quote(resource_id)}"
    
    # Construct the full URL
    full_url = cdn_base_url + path_part

    # Parse the URL to get the domain
    parsed_url = urllib.parse.urlparse(full_url)
    domain = parsed_url.netloc

    # Check if the domain is in the allowed list
    if domain not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"The domain '{domain}' from URL '{full_url}' is not an allowed CDN domain."
        )

    return full_url

if __name__ == '__main__':
    # Example Usage (matches the provided example)
    try:
        url = construct_external_resource_url(
            cdn_base_url="https://cdn.example.com",
            resource_id="abcd1234",
            resource_type="image"
        )
        print(f"Constructed URL: {url}") # Expected: https://cdn.example.com/image/abcd1234
    except ValueError as e:
        print(f"Error: {e}")

    # Example with a disallowed domain
    try:
        url = construct_external_resource_url(
            cdn_base_url="https://malicious.example.com",
            resource_id="badscript",
            resource_type="script"
        )
        print(f"Constructed URL: {url}")
    except ValueError as e:
        print(f"Error: {e}") # Expected: Error message about disallowed domain

    # Example with cdn_base_url ending with a slash
    try:
        url = construct_external_resource_url(
            cdn_base_url="https://assets.example.net/",
            resource_id="style.css",
            resource_type="stylesheet"
        )
        print(f"Constructed URL: {url}") # Expected: https://assets.example.net/stylesheet/style.css
    except ValueError as e:
        print(f"Error: {e}")
    
    # Example with special characters in resource_id or resource_type
    try:
        url = construct_external_resource_url(
            cdn_base_url="https://static.example.org",
            resource_id="file with spaces.jpg",
            resource_type="images/photos"
        )
        print(f"Constructed URL: {url}") # Expected: https://static.example.org/images%2Fphotos/file%20with%20spaces.jpg
    except ValueError as e:
        print(f"Error: {e}")
