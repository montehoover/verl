USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

def get_user_role(user_id):
    """
    Retrieves the role of a user given their user_id.

    Args:
        user_id: The ID of the user.

    Returns:
        The role of the user as a string, or None if the user is not found.
    """
    user = USERS.get(user_id)
    if user:
        return user.get("role")
    return None

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

def can_access_content(user_id, content_id):
    """
    Checks if a user has permission to access specific content.

    Args:
        user_id: The ID of the user.
        content_id: The ID of the content.

    Returns:
        True if access is allowed, False otherwise.
    """
    user_role = get_user_role(user_id)
    user_data = USERS.get(user_id)
    content_data = CONTENT.get(content_id)

    if not user_data or not content_data:
        return False  # User or content not found

    if user_role == "admin":
        return True  # Admins can access all content

    user_subscription_level = user_data.get("subscription_level")
    content_required_level = content_data.get("required_level")

    if not user_subscription_level or not content_required_level:
        return False # Subscription or required level missing

    if user_subscription_level == "premium":
        return True # Premium users can access all content (basic and premium)
    elif user_subscription_level == "basic":
        return content_required_level == "basic" # Basic users can only access basic content
    
    return False # Default to no access

def view_content(uid, cont_id):
    """
    Allows a user to view content if they have the necessary permissions.

    Args:
        uid: The ID of the user.
        cont_id: The ID of the content.

    Returns:
        The content's title if access is allowed, otherwise 'Access Denied'.
    """
    if can_access_content(uid, cont_id):
        content_item = CONTENT.get(cont_id)
        if content_item:
            return content_item.get("title")
        else:
            # This case should ideally not be reached if can_access_content passed
            # and content_id was valid, but as a safeguard:
            return "Content not found" 
    else:
        return "Access Denied"
