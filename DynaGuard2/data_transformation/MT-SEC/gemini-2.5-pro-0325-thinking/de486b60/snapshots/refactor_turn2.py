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

    Args:
        uid: The ID of the user attempting to access the content.
        cont_id: The ID of the content the user wants to access.

    Returns:
        The content's title if the user has the proper permissions, 
        otherwise 'Access Denied'.
    """
    user = USERS.get(uid)
    content = CONTENT.get(cont_id)

    # Guard clause: Handle non-existent user or content first
    if not user or not content:
        return "Access Denied"

    user_role = user.get("role")
    user_subscription_level = user.get("subscription_level")
    content_required_level = content.get("required_level")
    content_title = content.get("title")

    # Guard clause: Admin users have access to all content
    if user_role == "admin":
        return content_title

    # Guard clause: Premium users have access to premium and basic content
    if user_subscription_level == "premium":
        # Premium users can access content of level 'premium' or 'basic'.
        # Given the current CONTENT structure, this means they get access if the content exists.
        return content_title
    
    # Guard clause logic for Basic users:
    if user_subscription_level == "basic":
        if content_required_level == "basic":
            return content_title
        else:
            # Basic user attempting to access content not marked 'basic'
            return "Access Denied"
        
    # Fallback: Handles users with subscription levels other than admin (role), premium, or basic,
    # or any other case not explicitly granting access.
    return "Access Denied"

if __name__ == '__main__':
    # Example Usage:
    print(f"User 1 viewing Basic Content 1: {view_content(1, 100)}")  # Expected: Basic Content 1
    print(f"User 2 viewing Basic Content 1: {view_content(2, 100)}")  # Expected: Basic Content 1
    print(f"User 2 viewing Premium Content 1: {view_content(2, 200)}") # Expected: Access Denied
    print(f"User 4 viewing Premium Content 1: {view_content(4, 200)}") # Expected: Premium Content 1
    print(f"User 1 viewing Premium Content 2: {view_content(1, 201)}") # Expected: Premium Content 2
    print(f"User 3 viewing Premium Content 2: {view_content(3, 201)}") # Expected: Access Denied
    print(f"User 5 (non-existent) viewing Basic Content 1: {view_content(5, 100)}") # Expected: Access Denied
    print(f"User 1 viewing Content 300 (non-existent): {view_content(1, 300)}") # Expected: Access Denied
