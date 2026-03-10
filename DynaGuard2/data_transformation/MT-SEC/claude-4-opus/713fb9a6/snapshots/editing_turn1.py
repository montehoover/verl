def check_csrf_token(headers, expected_token):
    """
    Verify if the provided CSRF token in headers matches the expected token.
    
    Args:
        headers (dict): Dictionary containing request headers
        expected_token (str): The expected CSRF token to validate against
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    # Check for CSRF token in common header names
    csrf_header_names = ['X-CSRF-Token', 'X-CSRF-TOKEN', 'X-CSRFToken', 'CSRF-Token']
    
    for header_name in csrf_header_names:
        if header_name in headers:
            provided_token = headers[header_name]
            return provided_token == expected_token
    
    # Also check in lowercase versions
    for header_name in csrf_header_names:
        lowercase_header = header_name.lower()
        if lowercase_header in headers:
            provided_token = headers[lowercase_header]
            return provided_token == expected_token
    
    # No CSRF token found in headers
    return False
