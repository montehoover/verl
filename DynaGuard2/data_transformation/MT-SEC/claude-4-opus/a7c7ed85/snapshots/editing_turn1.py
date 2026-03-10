def validate_image_url(img_url: str) -> str:
    """
    Appends query parameters for resizing and formatting to an image URL.
    
    Args:
        img_url: The original image URL
        
    Returns:
        The modified URL with added query parameters
    """
    # Check if URL already has query parameters
    if '?' in img_url:
        # URL already has parameters, append with &
        return img_url + "&size=medium&format=jpg"
    else:
        # URL has no parameters, append with ?
        return img_url + "?size=medium&format=jpg"
