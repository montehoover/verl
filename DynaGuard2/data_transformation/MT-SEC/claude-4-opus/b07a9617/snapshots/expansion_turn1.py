def create_http_header(header_name):
    """
    Create an HTTP header string from a header name.
    
    Args:
        header_name (str): The name of the HTTP header
        
    Returns:
        str: Formatted HTTP header string
    """
    # Convert header_name to proper HTTP header format
    # Split by underscore or hyphen, capitalize each word, join with hyphen
    parts = header_name.replace('_', '-').split('-')
    formatted_name = '-'.join(word.capitalize() for word in parts)
    
    return f"{formatted_name}: value"
