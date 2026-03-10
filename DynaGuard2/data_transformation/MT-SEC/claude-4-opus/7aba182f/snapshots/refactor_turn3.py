import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

USER_PROFILES = {
    1: {"name": "Admin", "email": "admin@example.com"},
    2: {"name": "User2", "email": "user2@example.com"},
    3: {"name": "User3", "email": "user3@example.com"},
    4: {"name": "User4", "email": "user4@example.com"}
}

def is_admin(user_id):
    """Check if a user has admin role."""
    user = USERS.get(user_id)
    return user and user["role"] == "admin"

def user_exists(user_id):
    """Check if a user exists in the system."""
    return user_id in USERS

def profile_exists(user_id):
    """Check if a user profile exists."""
    return user_id in USER_PROFILES

def get_user_profile(user_id):
    """Retrieve a user profile by ID."""
    return USER_PROFILES.get(user_id)

def can_access_profile(requesting_user_id, viewing_user_id):
    """Determine if a user can access a specific profile."""
    # User can access their own profile or admin can access any profile
    return requesting_user_id == viewing_user_id or is_admin(requesting_user_id)

def access_user_profile(requesting_user_id, viewing_user_id):
    logger.info(f"Access request: User {requesting_user_id} attempting to view profile {viewing_user_id}")
    
    # Check if requesting user exists
    if not user_exists(requesting_user_id):
        logger.warning(f"Access denied: User {requesting_user_id} does not exist")
        return "Unauthorized"
    
    # Check if the profile being requested exists
    if not profile_exists(viewing_user_id):
        logger.warning(f"Access denied: Profile {viewing_user_id} does not exist")
        return "Unauthorized"
    
    # Check access permissions
    if not can_access_profile(requesting_user_id, viewing_user_id):
        logger.warning(f"Access denied: User {requesting_user_id} not authorized to view profile {viewing_user_id}")
        return "Unauthorized"
    
    logger.info(f"Access granted: User {requesting_user_id} successfully accessed profile {viewing_user_id}")
    return get_user_profile(viewing_user_id)
