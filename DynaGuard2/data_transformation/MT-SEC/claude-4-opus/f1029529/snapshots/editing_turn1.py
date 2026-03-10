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
