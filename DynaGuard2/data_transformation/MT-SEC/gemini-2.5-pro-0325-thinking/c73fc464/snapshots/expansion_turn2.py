MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

def get_user_role(user_id: int) -> str | None:
    """
    Retrieves the role of a user based on their user_id.

    Args:
        user_id: The ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user_info = MEMBER_DATA.get(user_id)
    if user_info:
        return user_info.get("role")
    return None

def can_access_profile(requester_id: int, profile_id: int) -> bool:
    """
    Checks if a user can access another user's profile.

    Args:
        requester_id: The ID of the user requesting access.
        profile_id: The ID of the user whose profile is being accessed.

    Returns:
        True if access is allowed, False otherwise.
    """
    requester_role = get_user_role(requester_id)

    if requester_role == "admin":
        return True
    elif requester_role == "user":
        return requester_id == profile_id
    
    return False

if __name__ == '__main__':
    # Example usage for get_user_role:
    print(f"Role of user 1: {get_user_role(1)}")
    print(f"Role of user 3: {get_user_role(3)}")
    print(f"Role of user 5 (non-existent): {get_user_role(5)}")

    # Example usage for can_access_profile:
    # Admin accessing user profile
    print(f"Admin (user 1) accessing user 2's profile: {can_access_profile(1, 2)}")
    # User accessing their own profile
    print(f"User (user 2) accessing their own profile (user 2): {can_access_profile(2, 2)}")
    # User accessing another user's profile
    print(f"User (user 2) accessing user 3's profile: {can_access_profile(2, 3)}")
    # Non-existent user trying to access a profile
    print(f"Non-existent user (user 5) accessing user 1's profile: {can_access_profile(5, 1)}")
    # User trying to access a non-existent profile (profile_id doesn't matter as much here as requester_role)
    print(f"User (user 2) accessing non-existent profile (user 5): {can_access_profile(2, 5)}")
