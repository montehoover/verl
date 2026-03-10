# Global data structure to track user roles
user_roles = {}

# Global data structure to track user subscription levels
user_subscriptions = {}

# Global data structure to track content requirements
content_requirements = {}

# User and content data structures
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

def can_access_content(user_id):
    """Check if a user can access content based on their role.
    
    Args:
        user_id (int): The ID of the user to check
        
    Returns:
        bool: True if the user is an admin, False otherwise
    """
    return user_roles.get(user_id) == 'admin'

def check_content_access(user_id, content_id):
    """Check if a user can access specific content based on their role and subscription level.
    
    Args:
        user_id (int): The ID of the user to check
        content_id (int): The ID of the content to check access for
        
    Returns:
        bool: True if the user can access the content, False otherwise
    """
    # Admins can access all content
    if user_roles.get(user_id) == 'admin':
        return True
    
    # Check subscription level
    user_subscription = user_subscriptions.get(user_id, 'free')
    required_subscription = content_requirements.get(content_id, 'free')
    
    # Define subscription hierarchy
    subscription_levels = {
        'free': 0,
        'basic': 1,
        'premium': 2
    }
    
    # User can access if their subscription level is equal or higher than required
    user_level = subscription_levels.get(user_subscription, 0)
    required_level = subscription_levels.get(required_subscription, 0)
    
    return user_level >= required_level

def fetch_content(given_user_id, target_content_id):
    """Fetch content based on user's role and subscription status.
    
    Args:
        given_user_id (int): The user trying to access the content
        target_content_id (int): The intended content
        
    Returns:
        str: The content title if permitted, otherwise 'Access Denied'
    """
    # Check if user exists
    if given_user_id not in USERS:
        return 'Access Denied'
    
    # Check if content exists
    if target_content_id not in CONTENT:
        return 'Access Denied'
    
    user = USERS[given_user_id]
    content = CONTENT[target_content_id]
    
    # Admins have unrestricted access
    if user['role'] == 'admin':
        return content['title']
    
    # Define subscription hierarchy
    subscription_levels = {
        'basic': 1,
        'premium': 2
    }
    
    # Check if user's subscription level meets content requirements
    user_level = subscription_levels.get(user['subscription_level'], 0)
    required_level = subscription_levels.get(content['required_level'], 0)
    
    if user_level >= required_level:
        return content['title']
    else:
        return 'Access Denied'
