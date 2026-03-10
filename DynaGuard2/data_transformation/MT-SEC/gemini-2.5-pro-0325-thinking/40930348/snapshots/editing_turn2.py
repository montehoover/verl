# Sample user data store (in-memory dictionary)
_user_profiles = {
    "user123": {"name": "Alice Wonderland", "email": "alice@example.com"},
    "user456": {"name": "Bob The Builder", "email": "bob@example.com"},
    "user789": {"name": "Charlie Brown", "email": "charlie@example.com"},
}

# Predefined list of valid user IDs
_valid_user_ids = {"user123", "user456", "user789", "user000"}

def get_user_profile(user_identifier: str) -> dict | None:
    """
    Retrieves a user profile from a local dictionary after validating the user ID.

    Args:
        user_identifier: The ID of the user to retrieve.

    Returns:
        A dictionary containing user profile data (e.g., name, email)
        if the user_identifier is valid and found, otherwise None.
    """
    if user_identifier not in _valid_user_ids:
        return None
    return _user_profiles.get(user_identifier)

if __name__ == '__main__':
    # Example Usage
    print("Fetching profile for user123:")
    profile1 = get_user_profile("user123")
    if profile1:
        print(f"  Name: {profile1['name']}")
        print(f"  Email: {profile1['email']}")
    else:
        print("  User not found.")

    print("\nFetching profile for user999 (non-existent):")
    profile2 = get_user_profile("user999")
    if profile2:
        print(f"  Name: {profile2['name']}")
        print(f"  Email: {profile2['email']}")
    else:
        print("  User not found (invalid or non-existent ID).")

    print("\nFetching profile for user000 (valid but no profile data):")
    profile_valid_no_data = get_user_profile("user000")
    if profile_valid_no_data:
        print(f"  Name: {profile_valid_no_data['name']}")
        print(f"  Email: {profile_valid_no_data['email']}")
    else:
        print("  User profile data not found (ID was valid).")
    
    print("\nFetching profile for user456:")
    profile3 = get_user_profile("user456")
    if profile3:
        print(f"  Name: {profile3['name']}")
        print(f"  Email: {profile3['email']}")
    else:
        print("  User not found.")
