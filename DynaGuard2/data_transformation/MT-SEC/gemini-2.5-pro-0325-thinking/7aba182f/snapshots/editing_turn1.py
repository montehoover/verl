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

if __name__ == '__main__':
    # Example Usage
    print(f"User 1 profile: {can_view_own_profile(1)}")
    print(f"User 2 profile: {can_view_own_profile(2)}")
    print(f"User 4 profile (non-existent): {can_view_own_profile(4)}")
