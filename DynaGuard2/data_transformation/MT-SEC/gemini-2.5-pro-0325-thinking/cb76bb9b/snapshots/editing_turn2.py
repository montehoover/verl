def build_resource_url(base_url: str, path: str, res_type: str) -> str:
    """
    Constructs a URL from a base URL, a path, and a resource type.
    The resource type is prepended to the path.

    Args:
        base_url: The base URL (e.g., "http://example.com").
        path: The path component (e.g., "/users").
        res_type: The type of resource (e.g., "image", "script").

    Returns:
        The full URL as a string.
    """
    # Ensure the path starts with a slash if it doesn't already,
    # and res_type doesn't end with one, to avoid double slashes.
    # Or, more simply, ensure res_type is part of the path correctly.
    # For now, let's assume path might or might not start with /
    # and res_type is just a prefix.
    # A more robust solution would use urllib.parse.urljoin or os.path.join logic.

    # Simple concatenation, assuming path starts with '/' if needed after res_type
    # e.g. path = "/actual_path" and res_type = "images" -> "/images/actual_path"
    # If path is "actual_path" and res_type is "images", this will be "imagesactual_path"
    # Let's refine this to ensure a slash between res_type and path.

    # Standardizing the path components
    # Remove leading slash from path if res_type is present, as res_type will form the first part.
    # Ensure res_type does not end with a slash and path does not start with one if both are present.
    
    # A simple approach:
    # If path starts with '/', prepend res_type directly.
    # If path does not start with '/', add a '/' between res_type and path.
    
    # Let's ensure there's exactly one slash between base_url and the rest,
    # and one slash between res_type and path.

    # Clean base_url: remove trailing slash
    if base_url.endswith('/'):
        base_url = base_url[:-1]

    # Clean path: remove leading slash
    if path.startswith('/'):
        path = path[1:]
        
    # Clean res_type: remove trailing slash
    if res_type.endswith('/'):
        res_type = res_type[:-1]
    
    # Construct the full path
    full_path = f"/{res_type}/{path}"
    
    return base_url + full_path
