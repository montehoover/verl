def construct_url(base_url: str, path: str) -> str:
    """
    Construct a URL by concatenating the base_url and path.

    This simple utility does not modify slashes; it simply joins the two strings.
    """
    return f"{base_url}{path}"
