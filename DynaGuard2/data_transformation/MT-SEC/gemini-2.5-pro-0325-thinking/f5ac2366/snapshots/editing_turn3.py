import urllib.parse

ALLOWED_CDN_DOMAINS = {'cdn.example.com', 'assets.example.net', 'static.example.org'}
ALLOWED_RESOURCE_TYPES = {"document", "video", "image", "audio"}

def construct_categorized_resource_url(base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Constructs the full URL for a categorized internal resource.

    Args:
        base_url: The base URL string (e.g., "https://example.com/api").
        resource_id: The resource identifier string (e.g., "123xyz").
        resource_type: The type of resource (e.g., "document", "video").
                       Must be one of the predefined categories.

    Returns:
        The full URL as a string (e.g., "https://example.com/api/document/123xyz").

    Raises:
        ValueError: If the resource_type is not one of the predefined categories.
    """
    if resource_type not in ALLOWED_RESOURCE_TYPES:
        raise ValueError(
            f"Invalid resource_type: '{resource_type}'. "
            f"Allowed types are: {', '.join(sorted(list(ALLOWED_RESOURCE_TYPES)))}."
        )

    # Ensure base_url does not end with a slash to avoid double slashes when joining,
    # and resource_id does not start with a slash.
    cleaned_base_url = base_url.rstrip('/')
    cleaned_resource_id = resource_id.lstrip('/')
    
    # Construct the URL
    return f"{cleaned_base_url}/{resource_type}/{cleaned_resource_id}"


def construct_external_resource_url(cdn_base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Constructs the full URL for an external resource from a CDN.

    Args:
        cdn_base_url: The base URL of the CDN (e.g., "https://cdn.example.com").
        resource_id: The identifier of the resource (e.g., "item123").
        resource_type: The type of resource (e.g., "image", "script").

    Returns:
        The full URL as a string (e.g., "https://cdn.example.com/image/item123").

    Raises:
        ValueError: If the cdn_base_url is not pointing to an allowed CDN domain.
    """
    parsed_base_url = urllib.parse.urlparse(cdn_base_url)
    if parsed_base_url.hostname not in ALLOWED_CDN_DOMAINS:
        raise ValueError(
            f"Invalid CDN base URL: '{cdn_base_url}'. "
            f"Domain '{parsed_base_url.hostname}' is not in the allowed list: "
            f"{', '.join(sorted(list(ALLOWED_CDN_DOMAINS)))}."
        )

    # Ensure cdn_base_url does not end with a slash and resource_id does not start with one.
    # resource_type will be a path segment.
    cleaned_cdn_base_url = cdn_base_url.rstrip('/')
    cleaned_resource_id = resource_id.lstrip('/')
    
    # Construct the path components
    path_segments = [resource_type, cleaned_resource_id]
    
    # Join the base URL with the path segments
    # urllib.parse.urljoin can be tricky with paths, so manual construction is safer here
    # for this specific pattern.
    # We ensure the path starts with a single slash if the base URL doesn't end with one.
    
    # Using f-string for clarity, similar to the previous function.
    # The path part will be /resource_type/resource_id
    full_url = f"{cleaned_cdn_base_url}/{resource_type}/{cleaned_resource_id}"
    
    # Validate the final URL's domain again, just in case (though construction logic should be sound)
    # This is more of a double-check on the construction logic itself.
    # However, the primary check is on cdn_base_url's domain.
    # For this function, the initial check on cdn_base_url's domain is sufficient as per requirements.

    return full_url
