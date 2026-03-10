from typing import Dict

# Global data structure tracking user roles by user_id.
USER_ROLES: Dict[int, str] = {}

def can_access_content(user_id: int) -> bool:
    """
    Returns True if the given user_id is associated with an admin role, else False.
    """
    role = USER_ROLES.get(user_id)
    return isinstance(role, str) and role.lower() == "admin"
