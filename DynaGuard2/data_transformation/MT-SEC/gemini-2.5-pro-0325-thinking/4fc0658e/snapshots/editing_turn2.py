# Global data structure for user roles
# Example: {user_id: role}
USER_ROLES = {
    1: 'admin',
    2: 'editor',
    3: 'viewer',
}

def can_access_content(user_id: int) -> bool:
    """
    Checks if a user can access any content based on their role.
    Returns True if the user is an admin, otherwise False.
    """
    role = USER_ROLES.get(user_id)
    return role == 'admin'

# Example data structures for subscription levels
# These can be more complex in a real system (e.g., stored in a database)
USER_SUBSCRIPTIONS = {
    1: 'premium',  # admin user
    2: 'basic',    # editor user
    3: 'free',     # viewer user
    4: 'premium',  # another user
}

# Content ID to required subscription level
CONTENT_SUBSCRIPTION_REQUIREMENTS = {
    101: 'free',
    102: 'basic',
    103: 'premium',
    104: 'admin_only', # Special case, only admins can access
}

# Define an order for subscription levels for comparison
SUBSCRIPTION_LEVEL_HIERARCHY = {
    'free': 0,
    'basic': 1,
    'premium': 2,
    # 'admin_only' is not a typical subscription level, handled by role check
}

def check_content_access(user_id: int, content_id: int) -> bool:
    """
    Checks if a user can access specific content based on their role and subscription level.
    Admins can access any content.
    Other users must meet the content's subscription requirement.
    """
    user_role = USER_ROLES.get(user_id)

    # Admins have access to all content
    if user_role == 'admin':
        return True

    required_subscription = CONTENT_SUBSCRIPTION_REQUIREMENTS.get(content_id)

    # If content is marked as 'admin_only' and user is not admin (checked above), deny access
    if required_subscription == 'admin_only':
        return False

    # If content has no specific subscription requirement, assume it's accessible by non-admins
    # (or handle as 'free' if that's the policy)
    if not required_subscription:
        # This policy might need adjustment: what if content_id is unknown?
        # For now, let's assume unknown content_id means no specific requirement, thus accessible.
        # Or, more strictly, if content_id is not in CONTENT_SUBSCRIPTION_REQUIREMENTS, deny access.
        # Let's be strict: if content requirement is not defined, deny access unless it's free.
        # For simplicity here, if not found, let's say it's not accessible unless it's implicitly free.
        # A better approach would be to ensure all content has a defined requirement.
        # For this example, let's assume if not in the map, it's not accessible by non-admins.
        return False # Or True if policy is "accessible if not restricted"

    user_subscription_level_str = USER_SUBSCRIPTIONS.get(user_id)
    if not user_subscription_level_str:
        # User has no defined subscription, assume lowest (e.g., 'free') or deny
        # For this example, let's assume 'free' if not specified
        user_subscription_level_str = 'free'

    user_level = SUBSCRIPTION_LEVEL_HIERARCHY.get(user_subscription_level_str, -1) # -1 for unknown levels
    required_level = SUBSCRIPTION_LEVEL_HIERARCHY.get(required_subscription, -1) # -1 for unknown levels

    # If for some reason the required subscription isn't in the hierarchy (e.g. 'admin_only' was not caught)
    # or user's level is not in hierarchy, deny access.
    if user_level == -1 or required_level == -1:
        return False

    return user_level >= required_level
