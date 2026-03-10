import hashlib

# Sample user data store
USER_DATA_STORE = {
    "user001": {
        "id": "user001",
        "name": "Alice Johnson",
        "email": "alice.johnson@example.com",
        "age": 28,
        "location": "New York, NY"
    },
    "user002": {
        "id": "user002",
        "name": "Bob Smith",
        "email": "bob.smith@example.com",
        "age": 35,
        "location": "San Francisco, CA"
    },
    "user003": {
        "id": "user003",
        "name": "Charlie Brown",
        "email": "charlie.brown@example.com",
        "age": 42,
        "location": "Chicago, IL"
    }
}


def retrieve_user_profile(user_id):
    """
    Retrieve user profile from the local data store.
    
    Args:
        user_id (str): The unique identifier of the user
        
    Returns:
        dict: The user profile data
        
    Raises:
        KeyError: If the user_id does not exist in the data store
    """
    if user_id not in USER_DATA_STORE:
        raise KeyError(f"User with ID '{user_id}' not found")
    
    return USER_DATA_STORE[user_id]


def validate_signature(data, signature):
    """
    Validate a signature by comparing it with the hash of the data.
    
    Args:
        data (str): The data to validate
        signature (str): The signature to compare against
        
    Returns:
        bool: True if the signature matches the data's hash, False otherwise
    """
    try:
        # Compute SHA256 hash of the data
        data_hash = hashlib.sha256(data.encode('utf-8')).hexdigest()
        
        # Compare the computed hash with the provided signature
        return data_hash == signature
    except Exception:
        # Handle any unsupported operations gracefully
        return False
