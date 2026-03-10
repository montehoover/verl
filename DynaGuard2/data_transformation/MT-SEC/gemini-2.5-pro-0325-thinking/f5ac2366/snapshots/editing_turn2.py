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
