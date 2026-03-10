import hashlib

# Sample user database
USER_DATABASE = {
    "user123": {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "location": "New York"
    },
    "user456": {
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "age": 25,
        "location": "San Francisco"
    },
    "user789": {
        "name": "Bob Johnson",
        "email": "bob.johnson@example.com",
        "age": 35,
        "location": "Chicago"
    }
}


def fetch_user_profile(user_id):
    """
    Fetch user profile from the local USER_DATABASE.
    
    Args:
        user_id (str): The ID of the user to fetch
        
    Returns:
        dict: The user profile data
        
    Raises:
        KeyError: If the user_id is not found in the database
    """
    if user_id not in USER_DATABASE:
        raise KeyError(f"User with ID '{user_id}' not found")
    
    return USER_DATABASE[user_id]


def verify_signature(data, signature):
    """
    Verify that the provided signature matches the hash of the data.
    
    Args:
        data (str): The data to verify
        signature (str): The signature to compare against
        
    Returns:
        bool: True if the signature matches, False otherwise
    """
    computed_hash = hashlib.sha256(data.encode()).hexdigest()
    return computed_hash == signature
