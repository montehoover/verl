import re

def append_user_header(user_header: str) -> str:
    """
    Insert a custom header into the output headers of a response.
    
    Args:
        user_header: str, representing the custom header value input by the user
        
    Returns:
        str, which contains the full header string including the custom header value
        
    Note:
        Does not explicitly raise exceptions, but errors such as 're.error' 
        could occur due to invalid inputs or regex issues.
    """
    # Create the header string with the custom header value
    return f"Custom-Header: {user_header}"
