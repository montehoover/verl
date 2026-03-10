"""Module for fetching and verifying user profiles from a remote service.

This module provides functionality to retrieve user profiles from a remote service
and verify their authenticity using cryptographic signatures.
"""

# Dictionary mapping user IDs to their known valid cryptographic signatures
# These signatures are used to verify the authenticity of fetched profile data
TRUSTED_USER_PROFILES = {
    "user_abc": "4e99a098d57cbf9c6f67f176106769fb",
    "user_xyz": "12e4b2f76650484030e4d447d0b5f723"
}


def get_profile_from_service(user_id: str):
    """Simulate fetching user profile data from a remote service.
    
    This is a mock function that simulates a remote service call. In production,
    this would make an actual HTTP request or similar network call.
    
    Args:
        user_id: The unique identifier of the user whose profile needs to be fetched.
        
    Returns:
        dict: A dictionary containing user profile information with 'name' and 'email' keys.
        
    Raises:
        ConnectionError: If the user_id is not found in the trusted profiles,
                        simulating a failed network request.
    """
    # Check if the user exists in our trusted profiles database
    if user_id in TRUSTED_USER_PROFILES:
        # Return mock profile data for valid users
        return {"name": "Test User", "email": f"{user_id}@example.com"}
    else:
        # Simulate a connection error for unknown users
        raise ConnectionError(f"Could not fetch profile for user_id {user_id}")


def fetch_user_profile(user_id: str, signature: str):
    """Retrieve a user profile from a remote service and verify its signature.
    
    This function fetches user profile data from a remote service and validates
    the data's authenticity by comparing the provided signature against known
    trusted signatures. This ensures the profile data hasn't been tampered with
    during transmission.
    
    Args:
        user_id: The unique identifier of the user whose profile needs to be fetched.
        signature: The cryptographic signature accompanying the user profile data.
                  This signature is used to verify the authenticity of the data.
    
    Returns:
        dict or None: The user profile data as a dictionary if the signature is valid.
                     Returns None if the signature verification fails, indicating
                     the data may have been tampered with or is not trusted.
    
    Raises:
        ConnectionError: If the user profile could not be fetched from the remote service.
                        This error is propagated from the underlying service call.
    """
    try:
        # Attempt to fetch the profile data from the remote service
        profile_data = get_profile_from_service(user_id)
    except ConnectionError:
        # Re-raise the ConnectionError to inform the caller about the network issue
        # This allows the caller to handle connection failures appropriately
        raise
    
    # Verify the cryptographic signature to ensure data authenticity
    # Check if: 1) the user_id exists in our trusted profiles
    #          2) the provided signature matches the known valid signature
    if user_id in TRUSTED_USER_PROFILES and TRUSTED_USER_PROFILES[user_id] == signature:
        # Signature is valid - return the verified profile data
        return profile_data
    else:
        # Signature verification failed - data cannot be trusted
        # Return None to indicate the profile data should not be used
        return None
