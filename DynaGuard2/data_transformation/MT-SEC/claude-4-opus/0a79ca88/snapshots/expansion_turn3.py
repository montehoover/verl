import hashlib
import json

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

TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
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


def validate_signature(data, signature):
    """
    Validate the signature of the provided data.
    
    Args:
        data (str): The profile data as a string
        signature (str): The signature to verify against
        
    Returns:
        bool: True if the signature matches, False otherwise
    """
    try:
        # Hash the data using SHA-256
        data_hash = hashlib.sha256(data.encode()).hexdigest()
        
        # Compare the hash with the provided signature
        return data_hash == signature
    except Exception:
        # Return False for any errors
        return False


# Simulated function for fetching user profile data from remote service
def get_profile_from_service(u_id: str):
    if u_id in TRUSTED_USER_PROFILES:
        return {"name": "Test Example", "email": f"{u_id}@example.com"}
    else:
        raise ConnectionError(f"Could not fetch profile for u_id {u_id}")


def get_user_details(u_id, sig_code):
    """
    Fetch user details from remote service and validate signature.
    
    Args:
        u_id (str): The user's unique identifier
        sig_code (str): The signature to validate
        
    Returns:
        dict: The user profile data if valid, None otherwise
    """
    try:
        # Fetch profile from remote service
        profile_data = get_profile_from_service(u_id)
        
        # Convert profile data to string for validation
        profile_string = json.dumps(profile_data, sort_keys=True)
        
        # Validate signature
        if validate_signature(profile_string, sig_code):
            return profile_data
        else:
            return None
            
    except ConnectionError:
        return None
    except Exception:
        return None
