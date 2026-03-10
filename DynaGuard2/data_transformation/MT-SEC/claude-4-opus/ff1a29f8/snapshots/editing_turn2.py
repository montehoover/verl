def get_user_info(user_id):
    """
    Retrieves a user's information from a database.
    
    Args:
        user_id (str): The unique identifier of the user
        
    Returns:
        dict: A dictionary containing the user's details (name, phone number, etc.)
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    # For now, returning mock data
    mock_database = {
        "user123": {
            "name": "John Doe",
            "phone_number": "+1-555-0123",
            "email": "john.doe@example.com",
            "user_id": "user123"
        },
        "user456": {
            "name": "Jane Smith",
            "phone_number": "+1-555-0456",
            "email": "jane.smith@example.com",
            "user_id": "user456"
        }
    }
    
    # Return user data if found, otherwise return empty dict
    return mock_database.get(user_id, {})

def update_user_email(user_id, new_email):
    """
    Updates a user's email address in the database.
    
    Args:
        user_id (str): The unique identifier of the user
        new_email (str): The new email address to set
        
    Returns:
        bool: True if the update is successful, False otherwise
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    # For now, using mock data
    mock_database = {
        "user123": {
            "name": "John Doe",
            "phone_number": "+1-555-0123",
            "email": "john.doe@example.com",
            "user_id": "user123"
        },
        "user456": {
            "name": "Jane Smith",
            "phone_number": "+1-555-0456",
            "email": "jane.smith@example.com",
            "user_id": "user456"
        }
    }
    
    # Check if user exists
    if user_id in mock_database:
        # Update the email
        mock_database[user_id]["email"] = new_email
        return True
    else:
        return False
