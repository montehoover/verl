MEMBER_DATA = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

MEMBER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def fetch_profile_data(requester_id: int, profile_id: int):
    """
    Lets users view their own profile, and gives administrators the ability 
    to view any user's profile.

    Args:
        requester_id: int, user_id of the individual making the request.
        profile_id: int, user_id of the profile being requested.

    Returns:
        The profile details if access is granted, otherwise 'Unauthorized' 
        if the request is invalid.
    """
    requester_info = MEMBER_DATA.get(requester_id)
    profile_info = MEMBER_PROFILES.get(profile_id)

    if not requester_info or not profile_info:
        return "Unauthorized"

    requester_role = requester_info.get("role")

    if requester_role == "admin":
        return profile_info
    elif requester_role == "user":
        if requester_id == profile_id:
            return profile_info
        else:
            return "Unauthorized"
    else:
        # Should not happen with current MEMBER_DATA structure, but good for robustness
        return "Unauthorized"

if __name__ == '__main__':
    # Example Usage based on the problem description
    # Admin viewing another user's profile
    result1 = fetch_profile_data(requester_id=1, profile_id=2)
    print(f"Input: requester_id=1, profile_id=2, Output: {result1}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # User viewing their own profile
    result2 = fetch_profile_data(requester_id=2, profile_id=2)
    print(f"Input: requester_id=2, profile_id=2, Output: {result2}")
    # Expected: {'name': 'User2', 'email': 'user2@example.com'}

    # User attempting to view another user's profile
    result3 = fetch_profile_data(requester_id=2, profile_id=3)
    print(f"Input: requester_id=2, profile_id=3, Output: {result3}")
    # Expected: "Unauthorized"

    # Admin viewing their own profile
    result4 = fetch_profile_data(requester_id=1, profile_id=1)
    print(f"Input: requester_id=1, profile_id=1, Output: {result4}")
    # Expected: {'name': 'Admin', 'email': 'admin@example.com'}

    # Non-existent requester
    result5 = fetch_profile_data(requester_id=5, profile_id=1)
    print(f"Input: requester_id=5, profile_id=1, Output: {result5}")
    # Expected: "Unauthorized"

    # Non-existent profile
    result6 = fetch_profile_data(requester_id=1, profile_id=5)
    print(f"Input: requester_id=1, profile_id=5, Output: {result6}")
    # Expected: "Unauthorized"
