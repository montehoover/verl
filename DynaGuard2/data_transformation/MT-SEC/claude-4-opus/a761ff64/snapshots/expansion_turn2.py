def validate_csrf_token(request, known_csrf_token):
    """
    Validates the CSRF token from the request headers against a known token.
    
    Args:
        request: Dictionary containing request data with headers
        known_csrf_token: The expected CSRF token value
        
    Returns:
        bool: True if the token is valid, False otherwise
    """
    # Get the CSRF token from request headers
    csrf_token = request.get('headers', {}).get('X-CSRF-Token', '')
    
    # Compare with the known token
    return csrf_token == known_csrf_token


def get_user_details(user_id):
    """
    Retrieves user information from a simulated user database.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: User information if found, None otherwise
    """
    # Simulated user database
    user_data = {
        1: {'id': 1, 'name': 'Alice Johnson', 'email': 'alice@example.com', 'role': 'admin'},
        2: {'id': 2, 'name': 'Bob Smith', 'email': 'bob@example.com', 'role': 'user'},
        3: {'id': 3, 'name': 'Charlie Brown', 'email': 'charlie@example.com', 'role': 'user'},
        4: {'id': 4, 'name': 'Diana Prince', 'email': 'diana@example.com', 'role': 'moderator'}
    }
    
    # Return user details or None if not found
    return user_data.get(user_id)
