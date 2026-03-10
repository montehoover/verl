def create_header(header_name: str, header_value: str) -> str:
    """
    Create an HTTP header string from name and value.
    
    Args:
        header_name: The name of the HTTP header
        header_value: The value of the HTTP header
        
    Returns:
        A formatted header string as 'Header-Name: Header-Value'
    """
    return f"{header_name}: {header_value}"
