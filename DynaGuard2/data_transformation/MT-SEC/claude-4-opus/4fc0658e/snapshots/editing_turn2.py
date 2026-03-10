# Global data structure to track user roles
user_roles = {}

# Global data structure to track user subscription levels
user_subscriptions = {}

# Global data structure to track content requirements
content_requirements = {}

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
