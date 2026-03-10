def build_resource_url(base_url: str, path: str, res_type: str) -> str:
    """
    Build a full URL by concatenating base_url with the resource type and path.

    This function performs simple string concatenation without adding/removing slashes.
    The resulting URL is base_url + res_type + path.
    """
    return f"{base_url}{res_type}{path}"
