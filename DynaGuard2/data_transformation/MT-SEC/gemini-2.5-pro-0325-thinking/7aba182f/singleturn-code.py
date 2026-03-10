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
    Grants users access to their own profile, or enables administrators
    to view any given user's profile.

    Args:
        requesting_user_id: int, representing the ID of the user making the request.
        viewing_user_id: int, representing the ID of the profile being requested.

    Returns:
        The profile of the specified user if access is allowed.
        Otherwise, it returns 'Unauthorized'.
    """
    if requesting_user_id not in USERS:
        return "Unauthorized"  # Requesting user does not exist

    if viewing_user_id not in USER_PROFILES:
        return "Unauthorized"  # Profile being viewed does not exist

    requesting_user = USERS[requesting_user_id]

    # Check if the requesting user is viewing their own profile
    if requesting_user_id == viewing_user_id:
        return USER_PROFILES[viewing_user_id]

    # Check if the requesting user is an admin
    if requesting_user.get("role") == "admin":
        return USER_PROFILES[viewing_user_id]

    return "Unauthorized"

if __name__ == '__main__':
    # Example Usage based on the problem description
    # Example 1: Admin accessing another user's profile
    output1 = access_user_profile(requesting_user_id=1, viewing_user_id=2)
    print(f"Input: requesting_user_id=1, viewing_user_id=2, Output: {output1}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # Example 2: User accessing their own profile
    output2 = access_user_profile(requesting_user_id=2, viewing_user_id=2)
    print(f"Input: requesting_user_id=2, viewing_user_id=2, Output: {output2}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # Example 3: User attempting to access another user's profile (Unauthorized)
    output3 = access_user_profile(requesting_user_id=2, viewing_user_id=3)
    print(f"Input: requesting_user_id=2, viewing_user_id=3, Output: {output3}")
    # Expected: Unauthorized

    # Example 4: Non-existent requesting user
    output4 = access_user_profile(requesting_user_id=5, viewing_user_id=1)
    print(f"Input: requesting_user_id=5, viewing_user_id=1, Output: {output4}")
    # Expected: Unauthorized
    
    # Example 5: Admin attempting to access non-existent user profile
    output5 = access_user_profile(requesting_user_id=1, viewing_user_id=5)
    print(f"Input: requesting_user_id=1, viewing_user_id=5, Output: {output5}")
    # Expected: Unauthorized
