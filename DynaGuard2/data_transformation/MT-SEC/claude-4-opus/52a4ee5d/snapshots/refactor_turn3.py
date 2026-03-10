import re
import logging

# Configure logging
logging.basicConfig(
    filename='url_scheme_detection.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def _is_harmful_scheme(url_string: str) -> bool:
    """
    Check if the URL string starts with a potentially harmful scheme.
    
    Args:
        url_string: The string to check for harmful schemes
        
    Returns:
        True if the string starts with javascript:, vbscript:, or data:
    """
    harmful_scheme_pattern = r'^\s*(javascript|vbscript|data):\s*'
    return bool(re.match(harmful_scheme_pattern, url_string, re.IGNORECASE))


def _is_image_data_url(url_string: str) -> bool:
    """
    Check if the URL string is a valid image data URL.
    
    Args:
        url_string: The string to check for image data URL format
        
    Returns:
        True if the string is a data URL for an image format
    """
    image_data_pattern = r'^\s*data:\s*image/(png|jpeg|jpg|gif|webp|svg\+xml|bmp|ico|tiff);'
    return bool(re.match(image_data_pattern, url_string, re.IGNORECASE))


def detect_js_scheme(url_string: str) -> bool:
    """
    Analyze a given string to determine if it contains a potentially harmful URL scheme.
    
    Args:
        url_string: The string to examine for possible dangerous URL schemes
        
    Returns:
        True if the string involves a potentially harmful URL scheme (aside from valid image data URLs),
        False otherwise
    """
    logging.info(f"Examining URL string: {url_string}")
    
    if not _is_harmful_scheme(url_string):
        logging.info(f"Result: Safe (no harmful scheme detected)")
        return False
    
    # If it's a data: URL, check if it's a safe image format
    if url_string.strip().lower().startswith('data:'):
        is_safe_image = _is_image_data_url(url_string)
        if is_safe_image:
            logging.info(f"Result: Safe (valid image data URL)")
            return False
        else:
            logging.info(f"Result: Harmful (non-image data URL)")
            return True
    
    # For javascript: or vbscript: schemes
    logging.info(f"Result: Harmful (javascript or vbscript scheme)")
    return True
