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

def access_content(current_user_id: int, content_id: int):
    """
    Checks if a user can access specific content.

    Args:
        current_user_id: The ID of the user attempting to access the content.
        content_id: The ID of the content to be accessed.

    Returns:
        The content data (dictionary) if access is allowed, 
        otherwise "Access Denied" or "Content not found".
    """
    user = USERS.get(current_user_id)

    if not user:
        return "Access Denied"  # Or "User not found"

    content_item = CONTENT.get(content_id)
    if not content_item:
        return "Content not found"

    # Admin access
    if user.get("role") == "admin":
        return content_item

    # Subscription level access
    user_subscription_level = user.get("subscription_level")
    content_required_level = content_item.get("required_level")

    if user_subscription_level == content_required_level:
        return content_item
    
    return "Access Denied"

if __name__ == '__main__':
    # Test cases
    print(f"Admin user 1 accessing content 100: {access_content(1, 100)}")
    print(f"Admin user 1 accessing content 200: {access_content(1, 200)}")
    print(f"Admin user 1 accessing non-existent content 999: {access_content(1, 999)}")

    print(f"Non-admin user 2 accessing content 100: {access_content(2, 100)}")
    print(f"Non-admin user 2 accessing content 200: {access_content(2, 200)}") # Should be denied

    print(f"Non-admin user 4 (premium) accessing content 100: {access_content(4, 100)}") # Should be denied
    print(f"Non-admin user 4 (premium) accessing content 200: {access_content(4, 200)}")
    
    print(f"Non-existent user 5 accessing content 100: {access_content(5, 100)}")
