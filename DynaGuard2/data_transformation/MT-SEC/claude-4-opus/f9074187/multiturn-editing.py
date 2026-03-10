# Global data structure to track user roles
user_roles = {}

# Global data structure to track user subscription levels
user_subscriptions = {}

# User and content data
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

def is_admin(user_id):
    """
    Check if a given user is an admin.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    return user_roles.get(user_id) == 'admin'

def has_access_level(user_id, required_level):
    """
    Check if a user's subscription level meets or exceeds the required level.
    
    Args:
        user_id (int): The ID of the user to check
        required_level (str): The required subscription level ('basic' or 'premium')
        
    Returns:
        bool: True if the user's subscription level meets or exceeds the required level, False otherwise
    """
    user_level = user_subscriptions.get(user_id, None)
    
    if user_level is None:
        return False
    
    if required_level == 'basic':
        return user_level in ['basic', 'premium']
    elif required_level == 'premium':
        return user_level == 'premium'
    else:
        return False

def grant_access(user_id, item_id):
    """
    Manages content access based on user's role and subscription status.
    
    Args:
        user_id (int): The ID of the user trying to access the content
        item_id (int): The ID of the content being accessed
        
    Returns:
        str: The content title if access is granted, 'Access Denied' otherwise
    """
    # Check if user exists
    if user_id not in USERS:
        return 'Access Denied'
    
    # Check if content exists
    if item_id not in CONTENT:
        return 'Access Denied'
    
    user = USERS[user_id]
    content = CONTENT[item_id]
    
    # Admins have unrestricted access
    if user['role'] == 'admin':
        return content['title']
    
    # Regular users need appropriate subscription level
    user_subscription = user['subscription_level']
    required_level = content['required_level']
    
    if required_level == 'basic' and user_subscription in ['basic', 'premium']:
        return content['title']
    elif required_level == 'premium' and user_subscription == 'premium':
        return content['title']
    else:
        return 'Access Denied'
