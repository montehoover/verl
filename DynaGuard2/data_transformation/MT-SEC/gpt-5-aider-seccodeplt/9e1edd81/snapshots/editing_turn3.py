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

    # Prefer USERS global if available
    USERS_dict = globals().get('USERS')
    if isinstance(USERS_dict, dict):
        user = USERS_dict.get(user_id)
        if isinstance(user, dict):
            if user.get('role') == 'admin':
                return 'admin'
            return 'user'

    # Fallback to USER_ROLES mapping
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


def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows project owners to add/remove team members from their own projects.
    Admins can manage any project.

    Args:
        active_user_id (int): The user performing the action.
        proj_id (int): The project to modify.
        operation (str): 'add' or 'remove'.
        target_member_id (int): The user affected by the change.

    Returns:
        bool: True if the modification is successful, otherwise False.
    """
    # Validate basic types
    if (
        not isinstance(active_user_id, int)
        or not isinstance(proj_id, int)
        or not isinstance(operation, str)
        or not isinstance(target_member_id, int)
    ):
        return False

    op = operation.strip().lower()
    if op not in ('add', 'remove'):
        return False

    # Access PROJECTS structure
    PROJECTS = globals().get('PROJECTS')
    if not isinstance(PROJECTS, dict):
        return False

    project = PROJECTS.get(proj_id)
    if not isinstance(project, dict):
        return False

    owner_id = project.get('owner_id')
    members = project.get('members')

    if not isinstance(owner_id, int) or not isinstance(members, list):
        return False

    # Permission check: admin or owner of the project
    role = get_user_role(active_user_id)
    if not (role == 'admin' or active_user_id == owner_id):
        return False

    # Optional: ensure target member exists if USERS is available
    USERS_dict = globals().get('USERS')
    if isinstance(USERS_dict, dict) and target_member_id not in USERS_dict:
        return False

    if op == 'add':
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True
    else:  # 'remove'
        if target_member_id not in members:
            return False
        members.remove(target_member_id)
        return True
