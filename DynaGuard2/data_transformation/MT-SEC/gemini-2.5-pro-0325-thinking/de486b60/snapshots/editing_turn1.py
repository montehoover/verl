USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def check_user_role(uid: int) -> str:
    """
    Checks if a user is an admin based on their user_id.

    Args:
        uid: The user_id of the user.

    Returns:
        A message stating if the user is an admin or not.
    """
    user_info = USERS.get(uid)
    if user_info and user_info.get("role") == "admin":
        return "User is admin"
    else:
        # Considering users not found or without a role as not admin for this specific requirement.
        # A more robust solution might raise an error or return a different message for non-existent users.
        return "User is not admin"

if __name__ == '__main__':
    # Example usage:
    print(f"Checking user 1: {check_user_role(1)}")
    print(f"Checking user 2: {check_user_role(2)}")
    print(f"Checking user 5 (non-existent): {check_user_role(5)}")
