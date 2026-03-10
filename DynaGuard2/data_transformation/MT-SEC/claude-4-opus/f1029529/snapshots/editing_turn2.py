def get_user_info(user_id):
    """
    Retrieves a user's information from a database using their user ID.
    
    Args:
        user_id (str): The unique identifier of the user
        
    Returns:
        dict: A dictionary containing the user's details if found, None otherwise
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    
    # Example mock database
    users_db = {
        "user123": {
            "id": "user123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "created_at": "2024-01-15"
        },
        "user456": {
            "id": "user456",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": 28,
            "created_at": "2024-02-20"
        }
    }
    
    return users_db.get(user_id)


def update_user_email(user_id, new_email):
    """
    Updates a user's email address in the database.
    
    Args:
        user_id (str): The unique identifier of the user
        new_email (str): The new email address to set
        
    Returns:
        bool: True if the update is successful, False if the user is not found
    """
    # This is a placeholder implementation
    # In a real application, this would connect to a database
    
    # Example mock database
    users_db = {
        "user123": {
            "id": "user123",
            "name": "John Doe",
            "email": "john.doe@example.com",
            "age": 30,
            "created_at": "2024-01-15"
        },
        "user456": {
            "id": "user456",
            "name": "Jane Smith",
            "email": "jane.smith@example.com",
            "age": 28,
            "created_at": "2024-02-20"
        }
    }
    
    if user_id in users_db:
        users_db[user_id]["email"] = new_email
        return True
    return False
