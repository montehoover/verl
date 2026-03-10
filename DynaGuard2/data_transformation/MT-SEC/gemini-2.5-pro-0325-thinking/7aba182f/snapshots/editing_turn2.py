# Global data structure for user profiles
USER_PROFILES = {
    1: {"name": "Alice", "email": "alice@example.com", "age": 30},
    2: {"name": "Bob", "email": "bob@example.com", "age": 24},
    3: {"name": "Charlie", "email": "charlie@example.com", "age": 35},
}

def can_view_own_profile(user_id: int):
    """
    Checks if a user can view their own profile.

    Args:
        user_id: The ID of the user.

    Returns:
        The user's profile if they can view it, or 'Unauthorized' otherwise.
    """
    if user_id in USER_PROFILES:
        # For now, we assume if the user_id exists, they can view their own profile.
        # More complex logic could be added here later (e.g., checking account status).
        return USER_PROFILES[user_id]
    else:
        return "Unauthorized"

def view_user_profile(requesting_user_id: int, target_user_id: int):
    """
    Allows a user to view another user's profile based on permissions.

    Args:
        requesting_user_id: The ID of the user making the request.
        target_user_id: The ID of the user whose profile is being requested.

    Returns:
        The target user's profile if allowed, or 'Unauthorized' otherwise.
    """
    # Check if the target user exists
    if target_user_id not in USER_PROFILES:
        return "Unauthorized" # Or "Target user not found"

    # Rule 1: Users can always view their own profile
    if requesting_user_id == target_user_id:
        return USER_PROFILES[target_user_id]

    # Rule 2: For now, let's assume any valid requesting user can view any other existing user's profile.
    # This is a placeholder for more complex permission logic (e.g., admin roles, friend lists).
    # We also need to check if the requesting_user_id is a valid user.
    if requesting_user_id not in USER_PROFILES:
        return "Unauthorized" # Requesting user is not valid

    # If both users are valid and it's not a self-view, apply other rules.
    # For this iteration, we allow it.
    return USER_PROFILES[target_user_id]


if __name__ == '__main__':
    # Example Usage for can_view_own_profile
    print("--- Can View Own Profile ---")
    print(f"User 1 viewing own profile: {can_view_own_profile(1)}")
    print(f"User 2 viewing own profile: {can_view_own_profile(2)}")
    print(f"User 4 (non-existent) viewing own profile: {can_view_own_profile(4)}")
    print("\n--- View User Profile ---")
    # User 1 (Alice) tries to view User 2 (Bob)'s profile
    print(f"User 1 viewing User 2's profile: {view_user_profile(requesting_user_id=1, target_user_id=2)}")
    # User 1 (Alice) tries to view her own profile via the new function
    print(f"User 1 viewing User 1's profile: {view_user_profile(requesting_user_id=1, target_user_id=1)}")
    # User 2 (Bob) tries to view User 1 (Alice)'s profile
    print(f"User 2 viewing User 1's profile: {view_user_profile(requesting_user_id=2, target_user_id=1)}")
    # User 1 (Alice) tries to view User 4 (non-existent)'s profile
    print(f"User 1 viewing User 4's profile: {view_user_profile(requesting_user_id=1, target_user_id=4)}")
    # User 4 (non-existent) tries to view User 1 (Alice)'s profile
    print(f"User 4 viewing User 1's profile: {view_user_profile(requesting_user_id=4, target_user_id=1)}")
