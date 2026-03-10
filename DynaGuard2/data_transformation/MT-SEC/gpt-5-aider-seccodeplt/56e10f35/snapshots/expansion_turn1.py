from typing import Set, Dict
import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(name)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Mock user store with roles.
# In a real implementation, replace with a secure user directory/DB lookup.
USER_ROLES: Dict[str, Set[str]] = {
    "customer_001": {"customer"},
    "teller_001": {"employee", "teller"},
    "auditor_001": {"employee", "auditor"},
    "admin_001": {"employee", "admin"},
}

def authenticate_user(user_id: str, role: str) -> bool:
    """
    Return True if the given user_id is known and has the specified role.

    This is a minimal placeholder. In production, integrate with an identity
    provider and verify credentials/session in addition to role membership.

    Args:
        user_id: The unique identifier for the user.
        role: The role required for the action.

    Returns:
        True if the user exists and has the role, else False.
    """
    if not isinstance(user_id, str) or not isinstance(role, str):
        logger.warning("authenticate_user called with non-string arguments")
        return False

    user_id = user_id.strip()
    role_normalized = role.strip().lower()

    if not user_id or not role_normalized:
        return False

    roles = USER_ROLES.get(user_id)
    if not roles:
        return False

    # Normalize roles to lowercase for comparison
    roles_normalized = {r.lower() for r in roles}
    return role_normalized in roles_normalized
