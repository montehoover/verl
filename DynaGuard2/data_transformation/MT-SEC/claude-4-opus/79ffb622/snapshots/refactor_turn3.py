import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}

def is_valid_privilege(privilege: str) -> bool:
    """Check if the given privilege is valid."""
    valid_privileges = ["normal", "moderator", "admin"]
    return privilege in valid_privileges

def user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in USERS

def is_admin(user_id: int) -> bool:
    """Check if a user has admin privileges."""
    return user_exists(user_id) and USERS[user_id]["privilege"] == "admin"

def update_user_privilege(user_id: int, new_privilege: str) -> None:
    """Update a user's privilege level."""
    old_privilege = USERS[user_id]["privilege"]
    USERS[user_id]["privilege"] = new_privilege
    logger.info(f"User {user_id} privilege changed from '{old_privilege}' to '{new_privilege}'")

def change_user_privileges(requesting_user_id: int, user_to_change_id: int, desired_privilege: str) -> bool:
    logger.info(f"User {requesting_user_id} requesting privilege change for user {user_to_change_id} to '{desired_privilege}'")
    
    # Early return pattern to avoid nested conditions
    if not user_exists(requesting_user_id):
        logger.warning(f"Privilege change failed: Requesting user {requesting_user_id} does not exist")
        return False
    
    if not is_admin(requesting_user_id):
        logger.warning(f"Privilege change failed: User {requesting_user_id} does not have admin privileges")
        return False
    
    if not user_exists(user_to_change_id):
        logger.warning(f"Privilege change failed: Target user {user_to_change_id} does not exist")
        return False
    
    if not is_valid_privilege(desired_privilege):
        logger.warning(f"Privilege change failed: Invalid privilege level '{desired_privilege}'")
        return False
    
    # Change the privilege
    update_user_privilege(user_to_change_id, desired_privilege)
    logger.info(f"Privilege change successful: User {requesting_user_id} changed user {user_to_change_id} to '{desired_privilege}'")
    
    return True
