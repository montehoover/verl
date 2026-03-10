USERS = {
    1: {"user_id": 1, "role": "admin", "subscription_level": "premium"},
    2: {"user_id": 2, "role": "user", "subscription_level": "basic"},
    3: {"user_id": 3, "role": "user", "subscription_level": "basic"},
    4: {"user_id": 4, "role": "user", "subscription_level": "premium"}
}

import logging

CONTENT = {
    100: {"title": "Basic Content 1", "required_level": "basic"},
    101: {"title": "Basic Content 2", "required_level": "basic"},
    200: {"title": "Premium Content 1", "required_level": "premium"},
    201: {"title": "Premium Content 2", "required_level": "premium"}
}

def access_content(current_user_id: int, content_id: int):
    """
    Allows users to access content based on their subscription level.
    Admins can access any content.
    Users can only access content that matches their subscription level,
    with premium users also having access to basic content.

    Args:
        current_user_id: The ID of the user attempting to access content.
        content_id: The ID of the content being accessed.

    Returns:
        The content data if the user is authorized, otherwise 'Access Denied'.
    """
    # Configure basic logging
    # In a real application, this would typically be done once at startup.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    user = USERS.get(current_user_id)
    content_item = CONTENT.get(content_id)

    # Guard clause: User or content not found
    if not user or not content_item:
        logger.info(f"Access Denied: User {current_user_id} or Content {content_id} not found.")
        return "Access Denied"

    user_role = user.get("role")
    user_subscription = user.get("subscription_level")
    content_required_level = content_item.get("required_level")

    # Guard clause: Admin access
    if user_role == "admin":
        logger.info(f"Access Granted: Admin User {current_user_id} accessed Content {content_id}.")
        return content_item

    # Guard clause: If not admin, must be a 'user' role to proceed
    if user_role != "user":
        logger.info(f"Access Denied: User {current_user_id} with role '{user_role}' attempted to access Content {content_id}. Invalid role.")
        return "Access Denied"

    # User role is "user", check subscription levels
    if user_subscription == "premium":
        # Premium users can access premium and basic content
        if content_required_level in ["premium", "basic"]:
            logger.info(f"Access Granted: Premium User {current_user_id} accessed Content {content_id} (required: {content_required_level}).")
            return content_item
    elif user_subscription == "basic":
        # Basic users can only access basic content
        if content_required_level == "basic":
            logger.info(f"Access Granted: Basic User {current_user_id} accessed Content {content_id} (required: {content_required_level}).")
            return content_item
    
    # If user is "user" but subscription conditions are not met
    logger.info(f"Access Denied: User {current_user_id} (Subscription: {user_subscription}) attempted to access Content {content_id} (Required: {content_required_level}). Subscription mismatch.")
    return "Access Denied"
