import hashlib

# Example local user database
LOCAL_USER_DATABASE = {
    "user123": {
        "user_id": "user123",
        "name": "John Doe",
        "email": "john.doe@example.com",
        "age": 30,
        "location": "New York"
    },
    "user456": {
        "user_id": "user456",
        "name": "Jane Smith",
        "email": "jane.smith@example.com",
        "age": 25,
        "location": "Los Angeles"
    },
    "user789": {
        "user_id": "user789",
        "name": "Bob Johnson",
        "email": "bob.johnson@example.com",
        "age": 35,
        "location": "Chicago"
    }
}


def get_local_user_profile(user_id):
    """
    Retrieve a user's profile from the local database.
    
    Args:
        user_id (str): The ID of the user to retrieve
        
    Returns:
        dict: The user's profile data
        
    Raises:
        KeyError: If the user_id is not found in the database
    """
    if user_id in LOCAL_USER_DATABASE:
        return LOCAL_USER_DATABASE[user_id]
    else:
        raise KeyError(f"User ID '{user_id}' not found in local database")


def validate_signature(data, signature):
    """
    Validate that the provided signature matches the hash of the data.
    
    Args:
        data (str): The data to validate
        signature (str): The signature to compare against
        
    Returns:
        bool: True if the signature matches the hash of the data, False otherwise
    """
    # Compute SHA-256 hash of the data
    hash_object = hashlib.sha256(data.encode())
    computed_hash = hash_object.hexdigest()
    
    # Compare the computed hash with the provided signature
    return computed_hash == signature
