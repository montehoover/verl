def construct_internal_resource_url(base_url: str, resource_id: str) -> str:
    """
    Constructs the full URL for an internal resource.

    Args:
        base_url: The base URL string.
        resource_id: The resource identifier string.

    Returns:
        The full URL as a string.
    """
    return base_url + resource_id
