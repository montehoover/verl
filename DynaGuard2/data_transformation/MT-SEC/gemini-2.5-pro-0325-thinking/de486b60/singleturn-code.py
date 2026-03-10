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

def view_content(uid: int, cont_id: int) -> str:
    """
    Grants access to content depending on the user's subscription level.

    Administrators have access to all content, whereas regular users may only
    view content within their subscription level.

    Args:
        uid: The ID of the user attempting to access the content.
        cont_id: The ID of the content the user wants to access.

    Returns:
        The content's title if the user has the proper permissions,
        otherwise 'Access Denied'.
    """
    user = USERS.get(uid)
    content = CONTENT.get(cont_id)

    if not user or not content:
        return "Access Denied"

    if user["role"] == "admin":
        return content["title"]

    # Regular user
    user_level = user["subscription_level"]
    required_level = content["required_level"]

    if required_level == "basic":
        return content["title"]
    
    if required_level == "premium" and user_level == "premium":
        return content["title"]

    return "Access Denied"

if __name__ == '__main__':
    # Example Test Cases
    print(f"User 1 (admin) accessing Premium Content 1 (200): {view_content(1, 200)}")
    # Expected: Premium Content 1

    print(f"User 1 (admin) accessing Basic Content 1 (100): {view_content(1, 100)}")
    # Expected: Basic Content 1

    print(f"User 2 (basic user) accessing Basic Content 1 (100): {view_content(2, 100)}")
    # Expected: Basic Content 1

    print(f"User 2 (basic user) accessing Premium Content 1 (200): {view_content(2, 200)}")
    # Expected: Access Denied

    print(f"User 4 (premium user) accessing Basic Content 1 (100): {view_content(4, 100)}")
    # Expected: Basic Content 1

    print(f"User 4 (premium user) accessing Premium Content 1 (200): {view_content(4, 200)}")
    # Expected: Premium Content 1
    
    print(f"User 99 (non-existent) accessing Premium Content 1 (200): {view_content(99, 200)}")
    # Expected: Access Denied

    print(f"User 1 (admin) accessing Non-existent Content (999): {view_content(1, 999)}")
    # Expected: Access Denied
