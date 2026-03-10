USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

# Define a hierarchy for subscription levels for comparison
SUBSCRIPTION_HIERARCHY = {
    "basic": 1,
    "premium": 2
    # Add more levels here if needed, e.g., "free": 0
}

def fetch_content(given_user_id: int, target_content_id: int) -> str:
    """
    Manages content access based on the user's role and subscription status.
    Admins have unrestricted access.
    Regular users can only access content matching their subscription level or lower.
    Returns the content title if permitted, otherwise 'Access Denied'.
    """
    user_data = USERS.get(given_user_id)
    content_data = CONTENT.get(target_content_id)

    if not user_data or not content_data:
        return "Access Denied"  # User or content not found

    user_role = user_data.get("role")
    user_subscription = user_data.get("subscription_level")
    content_title = content_data.get("title")
    required_level_str = content_data.get("required_level")

    if not all([user_role, user_subscription, content_title, required_level_str]):
        # Data integrity issue, one of the expected keys is missing
        return "Access Denied"

    if user_role == "admin":
        return content_title

    # For regular users, check subscription level
    user_level_val = SUBSCRIPTION_HIERARCHY.get(user_subscription)
    required_level_val = SUBSCRIPTION_HIERARCHY.get(required_level_str)

    if user_level_val is None or required_level_val is None:
        # Subscription level string not found in hierarchy
        return "Access Denied"

    if user_level_val >= required_level_val:
        return content_title
    else:
        return "Access Denied"

if __name__ == '__main__':
    # Example Usage:
    # Admin access
    print(f"User 1 (Admin) accessing Content 100 (Basic): {fetch_content(1, 100)}")
    print(f"User 1 (Admin) accessing Content 200 (Premium): {fetch_content(1, 200)}")

    # Basic user access
    print(f"User 2 (Basic Sub) accessing Content 100 (Basic): {fetch_content(2, 100)}")
    print(f"User 2 (Basic Sub) accessing Content 200 (Premium): {fetch_content(2, 200)}") # Denied

    # Premium user access
    print(f"User 4 (Premium Sub) accessing Content 100 (Basic): {fetch_content(4, 100)}")
    print(f"User 4 (Premium Sub) accessing Content 200 (Premium): {fetch_content(4, 200)}")

    # Non-existent user/content
    print(f"User 99 (Non-existent) accessing Content 100: {fetch_content(99, 100)}")
    print(f"User 1 (Admin) accessing Content 999 (Non-existent): {fetch_content(1, 999)}")

    # User with basic subscription trying to access premium content with a typo in content data
    CONTENT[202] = {"title": "Premium Content 3", "required_level": "super_premium"} # Unknown level
    print(f"User 2 (Basic Sub) accessing Content 202 (Super Premium): {fetch_content(2, 202)}")

    # User with an unknown subscription level
    USERS[5] = {"user_id": 5, "role": "user", "subscription_level": "unknown_sub"}
    print(f"User 5 (Unknown Sub) accessing Content 100 (Basic): {fetch_content(5, 100)}")
