import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

USERS = {
    1: {"user_id": 1, "role": "admin", "privilege": "admin"},
    2: {"user_id": 2, "role": "user", "privilege": "normal"},
    3: {"user_id": 3, "role": "user", "privilege": "normal"},
    4: {"user_id": 4, "role": "moderator", "privilege": "moderator"}
}


def _user_exists(user_id: int) -> bool:
    """Check if a user exists in the system."""
    return user_id in USERS


def _has_admin_privileges(user_id: int) -> bool:
    """Check if a user has admin privileges."""
    if not _user_exists(user_id):
        return False
    return USERS[user_id]["privilege"] == "admin"


def _is_valid_privilege_level(privilege_level: str) -> bool:
    """Validate if the privilege level is allowed."""
    valid_privilege_levels = ["normal", "moderator", "admin"]
    return privilege_level in valid_privilege_levels


def _get_role_for_privilege(privilege_level: str) -> str:
    """Map privilege level to corresponding role."""
    privilege_to_role_mapping = {
        "admin": "admin",
        "moderator": "moderator",
        "normal": "user"
    }
    return privilege_to_role_mapping.get(privilege_level, "user")


def _update_user_privilege_and_role(user_id: int, new_privilege: str) -> None:
    """Update user's privilege and corresponding role."""
    USERS[user_id]["privilege"] = new_privilege
    USERS[user_id]["role"] = _get_role_for_privilege(new_privilege)


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Update a user's privileges if the acting user has admin rights.
    
    Args:
        acting_user_id: The ID of the user attempting to change privileges
        affected_user_id: The ID of the user whose privilege is being altered
        new_role: The new privilege level designated for that user
        
    Returns:
        True if the privilege modification is successful, otherwise False
    """
    # Guard clause: Check if acting user has admin privileges
    if not _has_admin_privileges(acting_user_id):
        logger.warning(
            f"Privilege update denied: User {acting_user_id} lacks admin privileges "
            f"(attempted to modify user {affected_user_id})"
        )
        return False
    
    # Guard clause: Check if affected user exists
    if not _user_exists(affected_user_id):
        logger.error(
            f"Privilege update failed: User {affected_user_id} does not exist "
            f"(requested by admin {acting_user_id})"
        )
        return False
    
    # Guard clause: Check if new privilege level is valid
    if not _is_valid_privilege_level(new_role):
        logger.error(
            f"Privilege update failed: Invalid privilege level '{new_role}' "
            f"(admin {acting_user_id} attempted to set for user {affected_user_id})"
        )
        return False
    
    # Get current privilege for logging
    current_privilege = USERS[affected_user_id]["privilege"]
    
    # Update the user's privilege and role
    _update_user_privilege_and_role(affected_user_id, new_role)
    
    # Log successful privilege change
    logger.info(
        f"Privilege update successful: Admin {acting_user_id} changed user {affected_user_id}'s "
        f"privilege from '{current_privilege}' to '{new_role}'"
    )
    
    return True
