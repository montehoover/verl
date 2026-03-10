import re


def include_custom_header(custom_value: str) -> str:
    """
    Insert a custom header into the response headers with user-supplied value.
    
    This function creates an HTTP header string with a sanitized user-provided value.
    The value is sanitized to remove any control characters, carriage returns, 
    line feeds, and other characters that could break HTTP header formatting or 
    enable header injection attacks.
    
    Args:
        custom_value (str): The user-defined value for the custom header.
                           This value will be sanitized before inclusion.
    
    Returns:
        str: The full header string in the format "X-Custom-Header: <sanitized_value>".
    
    Raises:
        While no explicit exceptions are raised, improper input or regex-related 
        problems could cause 're.error' exceptions.
    
    Example:
        >>> include_custom_header("MyValue123")
        'X-Custom-Header: MyValue123'
        
        >>> include_custom_header("Bad\r\nValue")
        'X-Custom-Header: BadValue'
    """
    # Sanitize the custom value by removing control characters
    # This regex pattern matches:
    # - \r: carriage return
    # - \n: line feed
    # - \x00-\x1f: ASCII control characters (0-31)
    # - \x7f: DEL character (127)
    sanitized_value = re.sub(r'[\r\n\x00-\x1f\x7f]', '', custom_value)
    
    # Construct the complete header string
    header_string = f"X-Custom-Header: {sanitized_value}"
    
    return header_string
