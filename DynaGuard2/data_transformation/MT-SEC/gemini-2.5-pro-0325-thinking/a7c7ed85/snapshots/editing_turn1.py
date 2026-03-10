def validate_image_url(img_url: str) -> str:
    """
    Appends query parameters for resizing and formatting to an image URL.

    Args:
        img_url: The original image URL.

    Returns:
        The modified image URL with query parameters.
    """
    return f"{img_url}?size=medium&format=jpg"
