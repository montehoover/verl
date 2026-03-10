from urllib.parse import urlparse

def extract_domain(url):
    """
    Extract the domain from a URL string.
    
    Args:
        url (str): The URL to extract the domain from
        
    Returns:
        str: The domain part of the URL
    """
    parsed_url = urlparse(url)
    return parsed_url.netloc
