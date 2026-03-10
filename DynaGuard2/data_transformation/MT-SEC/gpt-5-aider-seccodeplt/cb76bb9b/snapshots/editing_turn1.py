def build_url(base_url: str, path: str) -> str:
    """
    Build a full URL by concatenating base_url and path.

    This function performs simple string concatenation without adding/removing slashes.
    """
    return f"{base_url}{path}"
