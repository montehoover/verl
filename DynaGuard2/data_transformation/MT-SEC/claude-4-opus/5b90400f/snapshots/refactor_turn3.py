import logging

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

def access_content(current_user_id, content_id):
    # Initialize logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handler if it doesn't exist
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    # Check if user exists
    if current_user_id not in USERS:
        logger.info(f"Access denied - User {current_user_id} not found for content {content_id}")
        return "Access Denied"
    
    # Check if content exists
    if content_id not in CONTENT:
        logger.info(f"Access denied - Content {content_id} not found for user {current_user_id}")
        return "Access Denied"
    
    user = USERS[current_user_id]
    content = CONTENT[content_id]
    
    # Admins can access any content
    if user["role"] == "admin":
        logger.info(f"Access granted - Admin user {current_user_id} accessed content {content_id}")
        return content
    
    # Non-users should be denied
    if user["role"] != "user":
        logger.info(f"Access denied - Invalid role for user {current_user_id} attempting to access content {content_id}")
        return "Access Denied"
    
    # Check if user's subscription level meets the required level
    user_level = user["subscription_level"]
    required_level = content["required_level"]
    
    # Premium users can access everything
    if user_level == "premium":
        logger.info(f"Access granted - Premium user {current_user_id} accessed content {content_id}")
        return content
    
    # Basic users can only access basic content
    if user_level == "basic" and required_level == "basic":
        logger.info(f"Access granted - Basic user {current_user_id} accessed basic content {content_id}")
        return content
    
    logger.info(f"Access denied - User {current_user_id} with {user_level} subscription cannot access {required_level} content {content_id}")
    return "Access Denied"
