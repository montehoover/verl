import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


def update_privileges(acting_user_id: int, affected_user_id: int, new_role: str) -> bool:
    """
    Update the role of affected_user_id to new_role if acting_user_id is an admin.

    Returns:
        True if the role is successfully modified, False otherwise (invalid IDs, missing USERS, or insufficient permissions).
    """
    users: Dict[int, Dict[str, Any]] | None = globals().get("USERS")  # type: ignore[assignment]
    if not isinstance(users, dict):
        logger.warning("Global USERS dictionary is not defined or invalid.")
        return False

    # Validate acting user
    acting_user = users.get(acting_user_id)
    if not isinstance(acting_user, dict):
        return False

    is_admin = acting_user.get("role") == "admin" or acting_user.get("privilege") == "admin"
    if not is_admin:
        return False

    # Validate affected user
    affected_user = users.get(affected_user_id)
    if not isinstance(affected_user, dict):
        return False

    if not isinstance(new_role, str):
        return False

    logger.info(
        "Admin %s attempting to modify role of user %s to '%s'",
        acting_user_id,
        affected_user_id,
        new_role,
    )

    affected_user["role"] = new_role
    return True
