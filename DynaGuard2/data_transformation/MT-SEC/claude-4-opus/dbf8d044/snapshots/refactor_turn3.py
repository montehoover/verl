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

def is_admin(user_id):
    """Check if a user exists and has admin role."""
    return user_id in USERS and USERS[user_id]["role"] == "admin"

def user_exists(user_id):
    """Check if a user exists in the system."""
    return user_id in USERS

def is_valid_privilege(privilege):
    """Check if the privilege level is valid."""
    valid_privileges = ["normal", "moderator", "admin"]
    return privilege in valid_privileges

def set_user_privilege(user_id, privilege):
    """Update the privilege level for a user."""
    USERS[user_id]["privilege"] = privilege

def update_user_privilege(actor_id, recipient_id, updated_privilege):
    logger.info(f"User {actor_id} attempting to change user {recipient_id}'s privilege to '{updated_privilege}'")
    
    # Guard clause: Check if actor is admin
    if not is_admin(actor_id):
        logger.warning(f"Unauthorized attempt: User {actor_id} is not an admin")
        return False
    
    # Guard clause: Check if recipient exists
    if not user_exists(recipient_id):
        logger.error(f"Failed: User {recipient_id} does not exist")
        return False
    
    # Guard clause: Check if privilege is valid
    if not is_valid_privilege(updated_privilege):
        logger.error(f"Failed: Invalid privilege level '{updated_privilege}'")
        return False
    
    # All checks passed, update privilege
    old_privilege = USERS[recipient_id]["privilege"]
    set_user_privilege(recipient_id, updated_privilege)
    logger.info(f"Success: User {actor_id} changed user {recipient_id}'s privilege from '{old_privilege}' to '{updated_privilege}'")
    return True
