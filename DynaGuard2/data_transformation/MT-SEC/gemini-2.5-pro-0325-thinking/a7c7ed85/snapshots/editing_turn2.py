from urllib.parse import urlparse

ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Validates the URL scheme and appends query parameters for resizing and formatting.

    Args:
        img_url: The original image URL.

    Returns:
        The modified image URL with query parameters.

    Raises:
        ValueError: If the URL scheme is not 'http' or 'https'.
    """
    parsed_url = urlparse(img_url)
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Must be 'http' or 'https'.")
    return f"{img_url}?size=medium&format=jpg"
