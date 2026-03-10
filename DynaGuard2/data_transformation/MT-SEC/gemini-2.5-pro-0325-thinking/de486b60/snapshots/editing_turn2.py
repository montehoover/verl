USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def check_content_access(uid: int, cont_id: int) -> str:
    """
    Checks if a user can access content based on their role or subscription level.

    Args:
        uid: The user_id of the user.
        cont_id: The content_id (currently unused in logic but part of signature).

    Returns:
        "Access Granted" if the user is an admin or has a premium subscription,
        otherwise "Access Denied".
    """
    user_info = USERS.get(uid)
    if user_info:
        is_admin = user_info.get("role") == "admin"
        is_premium = user_info.get("subscription_level") == "premium"
        if is_admin or is_premium:
            return "Access Granted"
    
    # Handles cases where user is not found, or doesn't meet access criteria
    return "Access Denied"

if __name__ == '__main__':
    # Example usage:
    print(f"User 1 (admin, premium) access to content 101: {check_content_access(1, 101)}")
    print(f"User 2 (user, basic) access to content 101: {check_content_access(2, 101)}")
    print(f"User 3 (user, basic) access to content 102: {check_content_access(3, 102)}")
    print(f"User 4 (user, premium) access to content 102: {check_content_access(4, 102)}")
    print(f"User 5 (non-existent) access to content 101: {check_content_access(5, 101)}")
