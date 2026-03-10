USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def access_user_profile(requesting_user_id: int, viewing_user_id: int):
    """
    Allows users to access their own profile, while administrators can view any user's profile.

    Args:
        requesting_user_id: The ID of the user making the request.
        viewing_user_id: The ID of the user whose profile is being requested.

    Returns:
        The profile of the specified user if access is allowed, otherwise 'Unauthorized'.
    """
    # Check if the requesting user is valid
    if requesting_user_id not in USERS:
        return "Unauthorized"

    # Check if the user whose profile is being viewed is valid and has a profile
    if viewing_user_id not in USERS or viewing_user_id not in USER_PROFILES:
        return "Unauthorized" # Or a more specific error like "Target user or profile not found"

    requesting_user = USERS[requesting_user_id]
    
    # Rule 1: Users can access their own profile
    if requesting_user_id == viewing_user_id:
        return USER_PROFILES[viewing_user_id]

    # Rule 2: Administrators can view any user's profile
    if requesting_user["role"] == "admin":
        return USER_PROFILES[viewing_user_id]

    # Otherwise, access is denied
    return "Unauthorized"

if __name__ == '__main__':
    # Test cases
    print("--- Access User Profile Tests ---")

    # Admin (user 1) viewing their own profile
    print(f"Admin (1) viewing own profile (1): {access_user_profile(1, 1)}")
    # Expected: {'name': 'Admin', 'email': 'admin@example.com'}

    # Admin (user 1) viewing another user's profile (user 2)
    print(f"Admin (1) viewing User2's profile (2): {access_user_profile(1, 2)}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # Regular user (user 2) viewing their own profile (user 2)
    print(f"User2 (2) viewing own profile (2): {access_user_profile(2, 2)}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # Regular user (user 2) viewing another user's profile (user 3) - Should be Unauthorized
    print(f"User2 (2) viewing User3's profile (3): {access_user_profile(2, 3)}")
    # Expected: Unauthorized

    # Regular user (user 2) viewing admin's profile (user 1) - Should be Unauthorized
    print(f"User2 (2) viewing Admin's profile (1): {access_user_profile(2, 1)}")
    # Expected: Unauthorized

    # Requesting user does not exist (user 5)
    print(f"Non-existent user (5) viewing User2's profile (2): {access_user_profile(5, 2)}")
    # Expected: Unauthorized

    # Viewing user does not exist (user 5)
    print(f"Admin (1) viewing non-existent user's profile (5): {access_user_profile(1, 5)}")
    # Expected: Unauthorized
    
    # Viewing user exists in USERS but not in USER_PROFILES (e.g. if we add user 5 to USERS but not USER_PROFILES)
    # To test this, we'd temporarily modify the dictionaries or add a new user.
    # For now, this case is covered by `viewing_user_id not in USER_PROFILES`
    USERS[5] = {"user_id": 5, "role": "user"} # Temporarily add user 5 without a profile
    print(f"Admin (1) viewing User5's profile (5, no profile data): {access_user_profile(1, 5)}")
    # Expected: Unauthorized
    del USERS[5] # Clean up
