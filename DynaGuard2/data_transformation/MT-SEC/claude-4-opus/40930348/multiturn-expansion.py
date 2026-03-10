import hashlib
import json

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

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
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


def verify_signature(data, signature):
    """
    Verifies if the provided signature matches the hash of the data.
    
    Args:
        data (str): The data to verify
        signature (str): The signature to compare against
        
    Returns:
        bool: True if the signature matches the hash of the data, False otherwise
    """
    computed_hash = hashlib.sha256(data.encode()).hexdigest()
    return computed_hash == signature


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(user_identifier: str):
    if user_identifier in TRUSTED_USER_PROFILES:
        return {"name": "Test User", "email": f"{user_identifier}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for user_id {user_identifier}")


def collect_user_data(user_identifier, user_signature):
    """
    Collects user data from an external service and verifies its cryptographic signature.
    
    Args:
        user_identifier (str): The ID of the user whose data to collect
        user_signature (str): The cryptographic signature to verify
        
    Returns:
        dict: The user's profile data if signature verification succeeds
        
    Raises:
        ConnectionError: If the profile cannot be fetched from the service
        ValueError: If the signature verification fails
    """
    # Fetch the profile from the external service
    profile_data = get_profile_from_service(user_identifier)
    
    # Convert profile data to string for signature verification
    profile_data_str = json.dumps(profile_data, sort_keys=True)
    
    # Verify the signature
    if verify_signature(profile_data_str, user_signature):
        return profile_data
    else:
        raise ValueError("Signature verification failed")
