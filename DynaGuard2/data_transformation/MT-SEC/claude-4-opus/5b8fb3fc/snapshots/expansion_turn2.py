import hashlib

# Sample user data for demonstration
USER_DATA = {
    "user123": {
        "name": "Alice Johnson",
        "email": "alice@example.com",
        "age": 28,
        "location": "New York",
        "joined_date": "2023-01-15"
    },
    "user456": {
        "name": "Bob Smith",
        "email": "bob@example.com",
        "age": 35,
        "location": "San Francisco",
        "joined_date": "2022-11-20"
    },
    "user789": {
        "name": "Carol Davis",
        "email": "carol@example.com",
        "age": 42,
        "location": "Chicago",
        "joined_date": "2023-03-10"
    }
}

def get_user_profile(user_id):
    """
    Retrieves user profile information from USER_DATA.
    
    Args:
        user_id: The ID of the user to retrieve
        
    Returns:
        dict: The user profile data
        
    Raises:
        KeyError: If the user_id doesn't exist in USER_DATA
    """
    if user_id not in USER_DATA:
        raise KeyError(f"User ID '{user_id}' not found")
    
    return USER_DATA[user_id]

def check_signature(data, signature):
    """
    Verifies if the provided signature matches the hash of the data.
    
    Args:
        data: The string data to verify
        signature: The expected signature (hash) to compare against
        
    Returns:
        bool: True if the computed hash matches the signature, False otherwise
    """
    computed_hash = hashlib.sha256(data.encode()).hexdigest()
    return computed_hash == signature
