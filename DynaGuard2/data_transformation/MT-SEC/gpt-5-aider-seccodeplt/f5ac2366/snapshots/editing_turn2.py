from typing import Set

ALLOWED_RESOURCE_TYPES: Set[str] = {"document", "video"}


def construct_categorized_resource_url(base_url: str, resource_id: str, resource_type: str) -> str:
    """
    Construct a full internal resource URL by concatenating base_url, resource_type, and resource_id.

    This function performs a simple string concatenation without any validation or
    normalization (e.g., it does not add or remove slashes other than the single slash
    between resource_type and resource_id).

    Args:
        base_url: The base URL as a string.
        resource_id: The resource identifier to append.
        resource_type: The category of the resource (must be one of ALLOWED_RESOURCE_TYPES).

    Returns:
        The concatenated URL string in the form: base_url + resource_type + "/" + resource_id.

    Raises:
        ValueError: If resource_type is not in ALLOWED_RESOURCE_TYPES.
    """
    if resource_type not in ALLOWED_RESOURCE_TYPES:
        raise ValueError(
            f"Unsupported resource_type: {resource_type}. Allowed types: {sorted(ALLOWED_RESOURCE_TYPES)}"
        )
    return f"{base_url}{resource_type}/{resource_id}"
