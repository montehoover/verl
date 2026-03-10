# Sample user data dictionary
USER_DATA = {
    "user123": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "location": "New York",
        "joined_date": "2023-01-15"
    },
    "user456": {
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "age": 28,
        "location": "San Francisco",
        "joined_date": "2023-03-22"
    },
    "user789": {
        "name": "Bob Johnson",
        "email": "bob.johnson@example.com",
        "age": 35,
        "location": "Chicago",
        "joined_date": "2022-11-08"
    }
}


def retrieve_user_profile(user_id):
    """
    Retrieve user profile data for a given user_id.
    
    Args:
        user_id (str): The unique identifier for the user
        
    Returns:
        dict: The user profile data
        
    Raises:
        KeyError: If the user_id doesn't exist in USER_DATA
    """
    if user_id not in USER_DATA:
        raise KeyError(f"User ID '{user_id}' not found")
    
    return USER_DATA[user_id]
