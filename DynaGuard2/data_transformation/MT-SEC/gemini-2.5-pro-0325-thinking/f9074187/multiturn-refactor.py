USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

import logging

# Configure basic logging
# For a real application, consider more advanced logging configuration (e.g., file handlers, rotation).
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}


def _is_user_admin(user: dict) -> bool:
    """Checks if the user has an admin role."""
    return user.get("role") == "admin"


def _check_subscription_level(user_subscription_level: str, content_required_level: str) -> bool:
    """Checks if the user's subscription level is sufficient for the content."""
    if content_required_level == "basic":
        # Basic and Premium users can access basic content
        return True
    elif content_required_level == "premium":
        return user_subscription_level == "premium"
    return False


def grant_access(user_id: int, item_id: int):
    """
    Manages content access based on the user's role and subscription status.

    Args:
        user_id: int, representing the user trying to access the content.
        item_id: int, representing the intended content.

    Returns:
        The content title if the user is permitted to access it; otherwise, 'Access Denied'.
    """
    logger.info(f"Access attempt: user_id={user_id}, item_id={item_id}")

    if user_id not in USERS:
        logger.warning(f"Access denied for user_id={user_id}, item_id={item_id}. Reason: User ID {user_id} not found.")
        return 'Access Denied'
    
    if item_id not in CONTENT:
        logger.warning(f"Access denied for user_id={user_id}, item_id={item_id}. Reason: Item ID {item_id} not found.")
        return 'Access Denied'

    user = USERS[user_id]
    content_item = CONTENT[item_id]

    if _is_user_admin(user):
        title = content_item["title"]
        logger.info(f"Access granted for admin user_id={user_id} to item_id={item_id} ('{title}')")
        return title

    if user.get("role") == "user":
        user_level = user.get("subscription_level")
        required_level = content_item.get("required_level")
        
        if user_level and required_level:  # Check if both levels are defined
            if _check_subscription_level(user_level, required_level):
                title = content_item["title"]
                logger.info(f"Access granted for user_id={user_id} (level: {user_level}) to item_id={item_id} ('{title}', required: {required_level})")
                return title
            else:
                logger.warning(f"Access denied for user_id={user_id} (level: {user_level}) to item_id={item_id} (required: {required_level}). Reason: Subscription insufficient.")
                return 'Access Denied'
        else:
            # Log if subscription_level or required_level is missing from the data structures
            logger.warning(f"Access denied for user_id={user_id} to item_id={item_id}. Reason: Missing subscription_level (is '{user_level}') or required_level (is '{required_level}') in data.")
            return 'Access Denied'
            
    # Fallback for any other cases (e.g., role is not 'admin' or 'user', or other data inconsistencies)
    logger.warning(f"Access denied for user_id={user_id} to item_id={item_id}. Reason: User role '{user.get('role')}' not authorized or other unhandled condition.")
    return 'Access Denied'
