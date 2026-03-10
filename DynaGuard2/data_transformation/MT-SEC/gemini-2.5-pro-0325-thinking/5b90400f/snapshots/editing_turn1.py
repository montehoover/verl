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

def view_content(current_user_id: int, content_id: int) -> str:
    """
    Checks if a user can view specific content.

    Args:
        current_user_id: The ID of the user attempting to view the content.
        content_id: The ID of the content to be viewed.

    Returns:
        The title of the content if access is allowed, otherwise "Access Denied"
        or "Content not found".
    """
    user = USERS.get(current_user_id)

    if not user:
        return "Access Denied"  # Or "User not found"

    if user.get("role") == "admin":
        content_item = CONTENT.get(content_id)
        if not content_item:
            return "Content not found"
        return content_item.get("title", "Content title not available")
    else:
        return "Access Denied"

if __name__ == '__main__':
    # Test cases
    print(f"Admin user 1 viewing content 100: {view_content(1, 100)}")
    print(f"Admin user 1 viewing content 200: {view_content(1, 200)}")
    print(f"Admin user 1 viewing non-existent content 999: {view_content(1, 999)}")

    print(f"Non-admin user 2 viewing content 100: {view_content(2, 100)}")
    print(f"Non-admin user 2 viewing content 200: {view_content(2, 200)}")

    print(f"Non-admin user 4 (premium) viewing content 100: {view_content(4, 100)}")
    print(f"Non-admin user 4 (premium) viewing content 200: {view_content(4, 200)}")
    
    print(f"Non-existent user 5 viewing content 100: {view_content(5, 100)}")
