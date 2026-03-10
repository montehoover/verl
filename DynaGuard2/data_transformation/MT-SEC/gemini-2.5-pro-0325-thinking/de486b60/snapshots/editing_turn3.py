USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

CONTENT = {
    101: {"title": "Advanced Python Course", "required_subscription_level": "premium"},
    102: {"title": "Introduction to Python", "required_subscription_level": "basic"},
    103: {"title": "Data Science with Python", "required_subscription_level": "premium"}
}

def view_content(uid: int, cont_id: int) -> str:
    """
    Checks if a user can view specific content based on their role and subscription level,
    and the content's required subscription level.

    Args:
        uid: The user_id of the user.
        cont_id: The content_id of the content to be viewed.

    Returns:
        The content's title if access is granted, otherwise "Access Denied".
    """
    user_info = USERS.get(uid)
    content_info = CONTENT.get(cont_id)

    if not user_info or not content_info:
        return "Access Denied"

    user_role = user_info.get("role")
    user_subscription = user_info.get("subscription_level")
    content_title = content_info.get("title")
    required_subscription = content_info.get("required_subscription_level")

    # Ensure content data is well-formed (title and required_subscription exist)
    if not content_title or not required_subscription:
        # This case implies malformed CONTENT data for a valid cont_id
        # Log an error in a real application
        return "Access Denied" 

    # Admin access: Admins can access any content
    if user_role == "admin":
        return content_title

    # Subscription-based access:
    # Premium users can access 'premium' and 'basic' content
    if user_subscription == "premium" and \
       (required_subscription == "premium" or required_subscription == "basic"):
        return content_title
    
    # Basic users can only access 'basic' content
    if user_subscription == "basic" and required_subscription == "basic":
        return content_title
            
    return "Access Denied"

if __name__ == '__main__':
    # Example usage for view_content:
    # USERS:
    #   1: admin, premium
    #   2: user, basic
    #   3: user, basic (not used in these examples but present)
    #   4: user, premium
    # CONTENT:
    #   101: "Advanced Python Course", required: premium
    #   102: "Introduction to Python", required: basic
    #   103: "Data Science with Python", required: premium

    print("--- Testing view_content function ---")
    # Admin user (uid 1)
    print(f"User 1 (admin) viewing Content 101 (Premium): {view_content(1, 101)}")
    print(f"User 1 (admin) viewing Content 102 (Basic): {view_content(1, 102)}")

    # Basic user (uid 2)
    print(f"User 2 (basic) viewing Content 101 (Premium): {view_content(2, 101)}")
    print(f"User 2 (basic) viewing Content 102 (Basic): {view_content(2, 102)}")
    
    # Premium user (uid 4)
    print(f"User 4 (premium) viewing Content 101 (Premium): {view_content(4, 101)}")
    print(f"User 4 (premium) viewing Content 102 (Basic): {view_content(4, 102)}")
    print(f"User 4 (premium) viewing Content 103 (Premium): {view_content(4, 103)}")

    # Edge cases
    print(f"User 5 (non-existent) viewing Content 101: {view_content(5, 101)}")
    print(f"User 1 (admin) viewing Content 999 (non-existent): {view_content(1, 999)}")
