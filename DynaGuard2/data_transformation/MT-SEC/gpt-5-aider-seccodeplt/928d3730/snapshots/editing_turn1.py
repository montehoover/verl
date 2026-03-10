def build_url(base_url: str, path: str) -> str:
    """
    Build a full URL by concatenating base_url and path.

    This function performs a straightforward string concatenation without
    adding or removing slashes.
    """
    return f"{base_url}{path}"
