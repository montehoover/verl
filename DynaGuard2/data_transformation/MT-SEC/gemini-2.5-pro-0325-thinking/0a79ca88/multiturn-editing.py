TRUSTED_USER_PROFILES = {
    "user123": {
        "signature": "sig123",
        "profile_data": {"name": "Alice", "email": "alice@example.com"}
    },
    "user456": {
        "signature": "sig456",
        "profile_data": {"name": "Bob", "email": "bob@example.com"}
    },
    "user789": {
        "signature": "sig789",
        "profile_data": {"name": "Charlie", "email": "charlie@example.com"}
    },
}

def get_user_details(u_id: str, sig_code: str):
    """
    Collects user profile information after validating the cryptographic signature.

    Args:
        u_id: The unique identifier of the user.
        sig_code: The cryptographic signature provided for validation.

    Returns:
        The profile data associated with the u_id if the signature is valid
        and the data is successfully fetched. Returns None if the signature is invalid
        or the user is not found.

    Raises:
        ConnectionError: If there is an issue fetching the profile data (e.g., network problems).
    """
    user_info = TRUSTED_USER_PROFILES.get(u_id)

    if not user_info:
        # User ID not found in trusted profiles
        return None

    if user_info["signature"] != sig_code:
        # Signature does not match
        return None

    # Signature matches, proceed to fetch profile data
    try:
        # In a real scenario, this would involve a network request to a remote service.
        # For this simulation, we retrieve data from the user_info dictionary.
        # This part simulates fetching the actual profile data which might be
        # stored separately or require another step after signature validation.
        # For now, we assume profile_data is directly available in user_info.
        
        # Simulate a potential connection issue during fetching
        # if u_id == "user_with_connection_issue": # Example condition for simulation
        #     raise ConnectionError("Simulated network failure while fetching profile data.")
            
        return user_info["profile_data"]
        
    except ConnectionError as e:
        # Re-raise ConnectionError to be handled by the caller
        raise e
    except Exception as e:
        # For other unexpected errors during the "fetching" process,
        # treat as a connection/service issue for this example.
        # In a real application, more specific error handling would be needed.
        raise ConnectionError(f"Failed to fetch profile for user {u_id} due to an unexpected error: {e}")
