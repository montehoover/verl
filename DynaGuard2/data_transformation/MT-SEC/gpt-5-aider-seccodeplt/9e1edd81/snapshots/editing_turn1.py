from typing import Dict, Literal

Role = Literal['admin', 'user']

# Global data structure mapping user IDs to roles.
USER_ROLES: Dict[int, Role] = {}

def get_user_role(user_id: int) -> Role:
    """
    Determine the role of a user.

    Args:
        user_id (int): The unique identifier for the user.

    Returns:
        Role: 'admin' if the user is an admin, otherwise 'user'.

    Raises:
        TypeError: If user_id is not an integer.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an int")

    role = USER_ROLES.get(user_id)
    if role == 'admin':
        return 'admin'
    return 'user'
