from typing import Dict, Literal, Set

Role = Literal['admin', 'user']

# Global data structure mapping user IDs to roles.
USER_ROLES: Dict[int, Role] = {}

# Global data structure mapping project IDs to sets of user IDs.
PROJECT_MEMBERSHIPS: Dict[int, Set[int]] = {}

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


def modify_project_membership(proj_id: int, operation: str, user_id: int) -> bool:
    """
    Modify the membership of a project by adding or removing a user.

    Args:
        proj_id (int): The unique identifier for the project.
        operation (str): Either 'add' or 'remove'.
        user_id (int): The user ID to add or remove from the project.

    Returns:
        bool: True if the operation changed membership, otherwise False.
    """
    if not isinstance(proj_id, int) or not isinstance(user_id, int) or not isinstance(operation, str):
        return False

    op = operation.lower().strip()
    if op == 'add':
        members = PROJECT_MEMBERSHIPS.setdefault(proj_id, set())
        if user_id in members:
            return False
        members.add(user_id)
        return True
    elif op == 'remove':
        members = PROJECT_MEMBERSHIPS.get(proj_id)
        if not members or user_id not in members:
            return False
        members.remove(user_id)
        if not members:
            PROJECT_MEMBERSHIPS.pop(proj_id, None)
        return True

    return False
