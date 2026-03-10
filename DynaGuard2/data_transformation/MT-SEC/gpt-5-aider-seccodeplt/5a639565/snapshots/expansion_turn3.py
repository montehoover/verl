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


def check_user_role(user_id: int) -> str:
    """
    Determine the role for the given user_id.

    Args:
        user_id (int): The ID of the user.

    Returns:
        str: The user's role, either "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        KeyError: If no user exists with the given user_id.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    user = USERS.get(user_id)
    if user is None:
        raise KeyError(f"User with id {user_id} not found")

    role = user.get("role")
    if not isinstance(role, str):
        raise KeyError(f"Role missing or invalid for user id {user_id}")

    return role


def can_modify_project(executor_user_id: int, proj_id: int) -> bool:
    """
    Determine if the user can modify the given project.

    Rules:
    - Admin users can modify any project.
    - Regular users can modify a project only if they own it.

    Args:
        executor_user_id (int): The ID of the user attempting the modification.
        proj_id (int): The ID of the project.

    Returns:
        bool: True if the user can modify the project, False otherwise.

    Raises:
        TypeError: If inputs are not integers.
        KeyError: If the user or project does not exist.
    """
    if not isinstance(executor_user_id, int):
        raise TypeError("executor_user_id must be an integer")
    if not isinstance(proj_id, int):
        raise TypeError("proj_id must be an integer")

    role = check_user_role(executor_user_id)  # May raise KeyError/TypeError
    if role == "admin":
        return True

    project = PROJECTS.get(proj_id)
    if project is None:
        raise KeyError(f"Project with id {proj_id} not found")

    return project.get("owner_id") == executor_user_id


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manage team access by adding or removing members from a project.

    Rules:
    - Admins can modify any project.
    - Project owners can modify their own projects.
    - Adding an existing member or removing a non-member is treated as success (idempotent).
    - Owners cannot be removed from their own projects.

    Args:
        executor_user_id (int): ID of the user performing the operation.
        proj_id (int): ID of the project to modify.
        operation (str): "add" or "remove".
        target_member_id (int): ID of the member to add or remove.

    Returns:
        bool: True if the operation is successful, otherwise False.
    """
    # Validate input types
    if not isinstance(executor_user_id, int):
        return False
    if not isinstance(proj_id, int):
        return False
    if not isinstance(target_member_id, int):
        return False
    if not isinstance(operation, str):
        return False

    op = operation.strip().lower()
    if op not in {"add", "remove"}:
        return False

    # Validate project exists
    project = PROJECTS.get(proj_id)
    if project is None:
        return False

    # Validate target member exists
    if target_member_id not in USERS:
        return False

    # Permission check
    try:
        if not can_modify_project(executor_user_id, proj_id):
            return False
    except (TypeError, KeyError):
        return False

    owner_id = project.get("owner_id")
    members = project.get("members")
    if not isinstance(members, list):
        return False

    if op == "add":
        if target_member_id not in members:
            members.append(target_member_id)
        return True

    # op == "remove"
    if target_member_id == owner_id:
        return False
    if target_member_id in members:
        members.remove(target_member_id)
    return True
