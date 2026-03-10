def construct_internal_resource_url(base_url: str, resource_id: str) -> str:
    """
    Construct a full internal resource URL by concatenating base_url and resource_id.

    This function performs a simple string concatenation without any validation or
    normalization (e.g., it does not add or remove slashes).

    Args:
        base_url: The base URL as a string.
        resource_id: The resource identifier to append.

    Returns:
        The concatenated URL string.
    """
    return f"{base_url}{resource_id}"
