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

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
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


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_ident: str):
    if user_ident in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_ident}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_ident {user_ident}")


def fetch_profile_info(user_ident, user_sig):
    """
    Fetch user profile securely by validating the signature.
    
    Args:
        user_ident (str): The user identifier
        user_sig (str): The signature to validate
        
    Returns:
        dict: The user profile if signature is valid
        
    Raises:
        ValueError: If the signature is invalid
        ConnectionError: If the profile cannot be fetched
    """
    # Check if user_ident is in trusted profiles
    if user_ident not in TRUSTED_USER_PROFILES:
        raise ValueError(f"User {user_ident} is not a trusted profile")
    
    # Get the expected signature for this user
    expected_signature = TRUSTED_USER_PROFILES[user_ident]
    
    # Validate the provided signature matches the expected one
    if user_sig != expected_signature:
        raise ValueError("Invalid signature")
    
    # Fetch and return the profile from the service
    return get_profile_from_service(user_ident)
