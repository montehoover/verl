from typing import Dict, Optional

# Global data structure tracking user roles by user_id.
USER_ROLES: Dict[int, str] = {}

# Global data structure tracking user subscription levels by user_id.
USER_SUBSCRIPTIONS: Dict[int, str] = {}

# Global data structure tracking required subscription levels by content_id.
CONTENT_REQUIRED_SUBSCRIPTION: Dict[int, str] = {}

# Provided global dictionaries with user and content details.
# These may be populated elsewhere in the application.
USERS: Dict[int, Dict[str, str]] = {}
CONTENT: Dict[int, Dict[str, str]] = {}

# Ordered mapping of subscription levels to comparable ranks.
SUBSCRIPTION_LEVELS_ORDER: Dict[str, int] = {
    "none": 0,
    "free": 1,
    "basic": 2,
    "standard": 3,
    "premium": 4,
    "vip": 5,
}

def _level_rank(level: Optional[str]) -> int:
    if not isinstance(level, str):
        return 0
    return SUBSCRIPTION_LEVELS_ORDER.get(level.lower(), 0)

def can_access_content(user_id: int) -> bool:
    """
    Returns True if the given user_id is associated with an admin role, else False.
    """
    role = USER_ROLES.get(user_id)
    return isinstance(role, str) and role.lower() == "admin"

def check_content_access(user_id: int, content_id: int) -> bool:
    """
    Returns True if the user can access the content based on role and subscription level.
    Admins can access all content.
    Non-admins must have a subscription level >= the content's required level.
    If content has no explicit requirement, it's treated as 'free'.
    """
    # Role-based access: admins can access anything
    if can_access_content(user_id):
        return True

    # Subscription-based access for non-admins
    user_level = USER_SUBSCRIPTIONS.get(user_id)
    required_level = CONTENT_REQUIRED_SUBSCRIPTION.get(content_id, "free")

    return _level_rank(user_level) >= _level_rank(required_level)

def fetch_content(given_user_id: int, target_content_id: int) -> str:
    """
    Returns the content title if the user is permitted to access it; otherwise, 'Access Denied'.
    Admins have unrestricted access. Regular users must have a subscription level
    greater than or equal to the content's required level.
    """
    user = USERS.get(given_user_id)
    content = CONTENT.get(target_content_id)

    # Validate existence of user and content
    if not user or not content:
        return "Access Denied"

    # Admins have unrestricted access
    role = user.get("role", "")
    if isinstance(role, str) and role.lower() == "admin":
        return content.get("title", "Access Denied")

    # Check subscription level for non-admins
    user_level = user.get("subscription_level")
    required_level = content.get("required_level", "free")

    if _level_rank(user_level) >= _level_rank(required_level):
        return content.get("title", "Access Denied")

    return "Access Denied"
