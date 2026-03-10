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

# Define a hierarchy for subscription levels
SUBSCRIPTION_HIERARCHY = {
    "basic": 1,
    "premium": 2
}

def can_access_level(user_id, required_level):
    """
    Determines if a user is authorized to access content.

    Args:
        user_id: The ID of the user.
        required_level: The subscription level required for the content.

    Returns:
        True if the user can access the content, False otherwise.
    """
    user = USERS.get(user_id)

    if not user:
        return False  # User not found

    # Admins have access to all content
    if user["role"] == "admin":
        return True

    # For regular users, check subscription level
    user_level = user["subscription_level"]

    # Check if the required level is valid
    if required_level not in SUBSCRIPTION_HIERARCHY:
        return False # Or raise an error for invalid required_level

    # Check if the user's level is valid
    if user_level not in SUBSCRIPTION_HIERARCHY:
        return False # Or raise an error for invalid user_level

    # User can access if their subscription level is greater than or equal to the required level
    return SUBSCRIPTION_HIERARCHY[user_level] >= SUBSCRIPTION_HIERARCHY[required_level]

def is_content_accessible(current_user_id, content_id):
    """
    Determines if content is accessible to a user.

    Args:
        current_user_id: The ID of the current user.
        content_id: The ID of the content.

    Returns:
        True if the content is accessible to the user, False otherwise.
    """
    content_item = CONTENT.get(content_id)

    if not content_item:
        return False  # Content not found

    required_level = content_item["required_level"]
    return can_access_level(current_user_id, required_level)

def access_content(current_user_id, content_id):
    """
    Checks if a user can access content and returns the title or 'Access Denied'.

    Args:
        current_user_id: The ID of the current user.
        content_id: The ID of the content.

    Returns:
        The content title if access is granted, otherwise 'Access Denied'.
    """
    if is_content_accessible(current_user_id, content_id):
        # is_content_accessible already checks if content_item exists
        content_item = CONTENT.get(content_id)
        return content_item["title"]
    else:
        return 'Access Denied'

if __name__ == '__main__':
    # Test cases for can_access_level
    print(f"User 1 (admin) access basic: {can_access_level(1, 'basic')}")  # Expected: True
    print(f"User 1 (admin) access premium: {can_access_level(1, 'premium')}") # Expected: True
    
    print(f"User 2 (basic user) access basic: {can_access_level(2, 'basic')}") # Expected: True
    print(f"User 2 (basic user) access premium: {can_access_level(2, 'premium')}") # Expected: False
    
    print(f"User 4 (premium user) access basic: {can_access_level(4, 'basic')}") # Expected: True
    print(f"User 4 (premium user) access premium: {can_access_level(4, 'premium')}") # Expected: True
    
    print(f"User 5 (non-existent) access basic: {can_access_level(5, 'basic')}") # Expected: False
    print(f"User 2 (basic user) access unknown_level: {can_access_level(2, 'unknown_level')}") # Expected: False

    print("\n# Test cases for is_content_accessible")
    # User 1 (admin)
    print(f"User 1 access content 100 (basic): {is_content_accessible(1, 100)}")  # Expected: True
    print(f"User 1 access content 200 (premium): {is_content_accessible(1, 200)}") # Expected: True

    # User 2 (basic user)
    print(f"User 2 access content 100 (basic): {is_content_accessible(2, 100)}")  # Expected: True
    print(f"User 2 access content 200 (premium): {is_content_accessible(2, 200)}") # Expected: False

    # User 4 (premium user)
    print(f"User 4 access content 100 (basic): {is_content_accessible(4, 100)}")  # Expected: True
    print(f"User 4 access content 200 (premium): {is_content_accessible(4, 200)}") # Expected: True

    # Non-existent user
    print(f"User 5 access content 100 (basic): {is_content_accessible(5, 100)}")  # Expected: False

    # Non-existent content
    print(f"User 1 access content 300 (non-existent): {is_content_accessible(1, 300)}") # Expected: False

    print("\n# Test cases for access_content")
    # User 1 (admin)
    print(f"User 1 accessing content 100 (basic): {access_content(1, 100)}")  # Expected: Basic Content 1
    print(f"User 1 accessing content 200 (premium): {access_content(1, 200)}") # Expected: Premium Content 1

    # User 2 (basic user)
    print(f"User 2 accessing content 100 (basic): {access_content(2, 100)}")  # Expected: Basic Content 1
    print(f"User 2 accessing content 200 (premium): {access_content(2, 200)}") # Expected: Access Denied

    # User 4 (premium user)
    print(f"User 4 accessing content 100 (basic): {access_content(4, 100)}")  # Expected: Basic Content 1
    print(f"User 4 accessing content 200 (premium): {access_content(4, 200)}") # Expected: Premium Content 1

    # Non-existent user
    print(f"User 5 accessing content 100 (basic): {access_content(5, 100)}")  # Expected: Access Denied

    # Non-existent content
    print(f"User 1 accessing content 300 (non-existent): {access_content(1, 300)}") # Expected: Access Denied
    
    # User with basic subscription trying to access premium content (already tested but good for access_content)
    print(f"User 3 (basic user) accessing content 201 (premium): {access_content(3, 201)}") # Expected: Access Denied
