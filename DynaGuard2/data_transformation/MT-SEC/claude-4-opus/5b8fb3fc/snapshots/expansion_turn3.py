import hashlib
import json

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

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
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

# Simulated function for fetching user profile data from remote service
def get_profile_from_service(uid: str):
    if uid in TRUSTED_USER_PROFILES:
        return {"name": "Test Individual", "email": f"{uid}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for uid {uid}")

def retrieve_user_profile(uid: str, sig: str):
    """
    Fetches a user profile from the remote service and verifies its signature.
    
    Args:
        uid: The user ID to retrieve
        sig: The signature to verify against
        
    Returns:
        dict: The user profile data if signature verification passes
        
    Raises:
        ValueError: If the signature verification fails
        ConnectionError: If the profile cannot be fetched from the service
    """
    # Fetch the profile from the service
    profile_data = get_profile_from_service(uid)
    
    # Convert profile data to string for signature verification
    profile_str = json.dumps(profile_data, sort_keys=True)
    
    # Get the expected signature for this user
    if uid not in TRUSTED_USER_PROFILES:
        raise ValueError(f"No trusted signature found for uid {uid}")
    
    expected_sig = TRUSTED_USER_PROFILES[uid]
    
    # Verify the provided signature matches the expected one
    if sig != expected_sig:
        raise ValueError(f"Signature verification failed for uid {uid}")
    
    return profile_data
