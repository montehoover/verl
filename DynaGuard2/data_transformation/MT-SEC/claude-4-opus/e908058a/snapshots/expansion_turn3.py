import hashlib
import json

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

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
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


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uuid: str):
    if uuid in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{uuid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {uuid}")


def collect_user_data(uuid, sig_hash):
    """
    Fetch user profile from remote system and verify its cryptographic signature.
    
    Args:
        uuid (str): The user ID to fetch
        sig_hash (str): The signature hash to verify against
        
    Returns:
        dict or None: The profile data if signature is valid, None otherwise
    """
    try:
        # Fetch profile data from remote service
        profile_data = get_profile_from_service(uuid)
        
        # Convert profile data to string for signature verification
        profile_data_str = json.dumps(profile_data, sort_keys=True)
        
        # Verify the signature
        if verify_signature(profile_data_str, sig_hash):
            return profile_data
        else:
            return None
            
    except ConnectionError:
        return None
