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

def view_content(uid, cont_id):
    """
    Determines if a user has permission to view specific content based on their subscription level.
    
    This function checks whether a user (identified by uid) has the necessary permissions
    to access content (identified by cont_id). Access is granted based on the user's role
    and subscription level compared to the content's required access level.
    
    Parameters:
    -----------
    uid : int
        The user ID of the user attempting to access the content.
        Must correspond to a valid user in the USERS dictionary.
    
    cont_id : int
        The content ID of the content the user wants to access.
        Must correspond to valid content in the CONTENT dictionary.
    
    Returns:
    --------
    str
        The title of the content if the user has proper permissions,
        otherwise returns 'Access Denied'.
    
    Access Rules:
    -------------
    - Admin users can access all content regardless of subscription level
    - Premium subscription users can access both basic and premium content
    - Basic subscription users can only access basic content
    - Invalid user IDs or content IDs result in access denial
    """
    
    # Guard clause: Verify that the user ID exists in our user database
    if uid not in USERS:
        return "Access Denied"
    
    # Guard clause: Verify that the content ID exists in our content database
    if cont_id not in CONTENT:
        return "Access Denied"
    
    # Retrieve user and content information from the dictionaries
    user = USERS[uid]
    content = CONTENT[cont_id]
    
    # Guard clause: Admin users have unrestricted access to all content
    if user["role"] == "admin":
        return content["title"]
    
    # Guard clause: Premium subscription users can access all content types
    if user["subscription_level"] == "premium":
        return content["title"]
    
    # Check if basic subscription user is trying to access basic content
    # Basic users are only allowed to view content marked as "basic"
    if user["subscription_level"] == "basic" and content["required_level"] == "basic":
        return content["title"]
    
    # If none of the above conditions are met, deny access
    return "Access Denied"
