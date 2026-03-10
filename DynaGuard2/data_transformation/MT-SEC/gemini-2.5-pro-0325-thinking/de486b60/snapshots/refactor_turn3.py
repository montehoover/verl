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
    Determines if a user has permission to view specific content based on their
    subscription level and role.

    The function checks against predefined USERS and CONTENT dictionaries.
    Admins have universal access. Premium users can access premium and basic
    content. Basic users can only access basic content. If the user or content
    ID is invalid, or if permissions are insufficient, access is denied.

    Args:
        uid (int): The unique identifier of the user attempting to access the content.
        cont_id (int): The unique identifier of the content the user wishes to access.

    Returns:
        str: The title of the content if the user has sufficient permissions.
             Returns "Access Denied" if the user or content is not found, or
             if the user does not have the required subscription level.
    """
    # Retrieve user and content details from the global dictionaries.
    user = USERS.get(uid)
    content = CONTENT.get(cont_id)

    # Guard clause: If the user or content ID is invalid, deny access immediately.
    if not user or not content:
        return "Access Denied"  # User or content not found.

    # Extract relevant details for permission checking.
    user_role = user.get("role")
    user_subscription_level = user.get("subscription_level")
    content_required_level = content.get("required_level")
    content_title = content.get("title")

    # Permission check: Admin users have unrestricted access to all content.
    if user_role == "admin":
        return content_title  # Admin access granted.

    # Permission check: Premium users can access both 'premium' and 'basic' content.
    if user_subscription_level == "premium":
        # Since premium users can access all defined content types,
        # and we've already confirmed the content exists, grant access.
        return content_title  # Premium access granted.
    
    # Permission check: Basic users can only access 'basic' content.
    if user_subscription_level == "basic":
        if content_required_level == "basic":
            return content_title  # Basic user access to basic content granted.
        else:
            # Basic user attempting to access content above their subscription level (e.g., premium).
            return "Access Denied"  # Basic user access to non-basic content denied.
        
    # Fallback: If none of the above conditions are met (e.g., user has an unrecognized
    # subscription level or role that isn't admin), deny access.
    return "Access Denied"  # Default denial for unhandled cases.

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
