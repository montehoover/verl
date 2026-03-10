from urllib.parse import urlparse

ALLOWED_SCHEMES = ["http", "https"]

def validate_image_url(img_url: str) -> str:
    """
    Appends query parameters for resizing and formatting to an image URL.
    
    Args:
        img_url: The original image URL
        
    Returns:
        The modified URL with added query parameters
        
    Raises:
        ValueError: If the URL scheme is not http or https
    """
    # Parse the URL to check the scheme
    parsed_url = urlparse(img_url)
    
    # Check if the scheme is allowed
    if parsed_url.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"Unsupported URL scheme: {parsed_url.scheme}. Only http and https are allowed.")
    
    # Check if URL already has query parameters
    if '?' in img_url:
        # URL already has parameters, append with &
        return img_url + "&size=medium&format=jpg"
    else:
        # URL has no parameters, append with ?
        return img_url + "?size=medium&format=jpg"
