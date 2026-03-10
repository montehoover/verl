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
    parsed_base_url = urllib.parse.urlparse(cdn_base_url)
    if parsed_base_url.netloc not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Domain '{parsed_base_url.netloc}' is not an allowed CDN domain. "
            f"Allowed domains are: {', '.join(ALLOWED_CDN_DOMAINS)}"
        )

    # Ensure cdn_base_url ends with a slash for proper joining
    if not cdn_base_url.endswith('/'):
        cdn_base_url += '/'

    # Construct path segments carefully to avoid double slashes if resource_type or resource_id might be empty
    # or already start with a slash. For simplicity, assuming they are simple identifiers.
    # A more robust solution might involve os.path.join-like behavior for URL paths.
    
    # Using urllib.parse.urljoin for robust path construction
    # First join base with resource_type
    path_part_one = urllib.parse.urljoin(cdn_base_url, resource_type)
    # Ensure the result of the first join has a trailing slash before joining the resource_id
    if not path_part_one.endswith('/'):
        path_part_one += '/'
    
    full_url = urllib.parse.urljoin(path_part_one, resource_id)
    
    return full_url
