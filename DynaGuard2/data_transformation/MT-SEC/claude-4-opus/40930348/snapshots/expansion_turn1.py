# Sample user database
USER_DATABASE = {
    "user001": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
        "location": "New York"
    },
    "user002": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 35,
        "location": "Los Angeles"
    },
    "user003": {
        "name": "Charlie Brown",
        "email": "charlie@example.com",
        "age": 42,
        "location": "Chicago"
    }
}


def fetch_user_profile(user_id):
    """
    Fetches a user profile from the USER_DATABASE.
    
    Args:
        user_id (str): The ID of the user to fetch
        
    Returns:
        dict: The user's profile data
        
    Raises:
        KeyError: If the user_id doesn't exist in the database
    """
    if user_id not in USER_DATABASE:
        raise KeyError(f"User with ID '{user_id}' not found")
    
    return USER_DATABASE[user_id]
