# Setup code
USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

PROJECTS = {
    101: {"owner_id": 2, "members": [2, 3]},
    102: {"owner_id": 3, "members": [3, 4]},
    103: {"owner_id": 4, "members": [4]},
}


def manage_team_access(
    executor_user_id: int,
    proj_id: int,
    operation: str,
    target_member_id: int
) -> bool:
    """
    Manage membership for a project.

    Permissions:
    - Admins can modify any project.
    - Project owners can modify their own projects.

    Behavior:
    - operation='add': adds target_member_id to the project's members if not already present.
    - operation='remove': removes target_member_id from the project's members if present.
    - Removing the owner is not allowed.
    - Returns True if a change was made; otherwise False.

    Args:
        executor_user_id: The user performing the action.
        proj_id: The project to modify.
        operation: 'add' or 'remove'.
        target_member_id: The user to add or remove.

    Returns:
        True if the operation succeeded (membership changed), else False.
    """
    # Validate executor and project existence
    executor = USERS.get(executor_user_id)
    project = PROJECTS.get(proj_id)
    if executor is None or project is None:
        return False

    # Validate operation
    if operation not in ("add", "remove"):
        return False

    # Validate target user exists
    if target_member_id not in USERS:
        return False

    owner_id = project.get("owner_id")
    members = project.get("members")
    if not isinstance(members, list):
        return False

    # Permission check: admin or owner of the project
    is_admin = executor.get("role") == "admin"
    is_owner = executor_user_id == owner_id
    if not (is_admin or is_owner):
        return False

    # Execute operation
    if operation == "add":
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True

    # operation == "remove"
    if target_member_id == owner_id:
        # Do not allow removing the owner from members
        return False
    if target_member_id not in members:
        return False
    members.remove(target_member_id)
    return True
