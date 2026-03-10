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

if __name__ == '__main__':
    # Test cases
    print(f"User 1 (admin) access basic: {can_access_level(1, 'basic')}")  # Expected: True
    print(f"User 1 (admin) access premium: {can_access_level(1, 'premium')}") # Expected: True
    
    print(f"User 2 (basic user) access basic: {can_access_level(2, 'basic')}") # Expected: True
    print(f"User 2 (basic user) access premium: {can_access_level(2, 'premium')}") # Expected: False
    
    print(f"User 4 (premium user) access basic: {can_access_level(4, 'basic')}") # Expected: True
    print(f"User 4 (premium user) access premium: {can_access_level(4, 'premium')}") # Expected: True
    
    print(f"User 5 (non-existent) access basic: {can_access_level(5, 'basic')}") # Expected: False
    print(f"User 2 (basic user) access unknown_level: {can_access_level(2, 'unknown_level')}") # Expected: False
