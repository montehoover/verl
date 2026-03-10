import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO)


def update_privileges(acting_user_id: int, affected_user_id: int) -> bool:
    """
    Check if acting_user_id corresponds to an admin in the global USERS dictionary.
    If admin, log the attempt to modify the affected user's role and return True.
    Return False otherwise.
    """
    users: Dict[int, Dict[str, Any]] | None = globals().get("USERS")  # type: ignore[assignment]
    if not isinstance(users, dict):
        logger.warning("Global USERS dictionary is not defined or invalid.")
        return False

    acting_user = users.get(acting_user_id)
    if not isinstance(acting_user, dict):
        return False

    is_admin = acting_user.get("role") == "admin" or acting_user.get("privilege") == "admin"

    if is_admin:
        logger.info(
            "Admin %s attempting to modify role of user %s",
            acting_user_id,
            affected_user_id,
        )
        return True

    return False
