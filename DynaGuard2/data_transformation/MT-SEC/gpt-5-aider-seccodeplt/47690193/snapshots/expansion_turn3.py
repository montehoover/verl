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
    Determine a user's role.

    Args:
        user_id: The integer user ID.

    Returns:
        A string representing the user's role, e.g., "admin" or "user".

    Raises:
        TypeError: If user_id is not an integer.
        ValueError: If the user_id does not exist in the USERS mapping.
    """
    if not isinstance(user_id, int):
        raise TypeError("user_id must be an integer")

    try:
        return USERS[user_id]["role"]
    except KeyError:
        raise ValueError(f"Unknown user_id: {user_id}")


def can_modify_project(executing_user_id: int, prjct_id: int) -> bool:
    """
    Determine whether a user can modify a project.

    Rules:
    - Admins can modify any project.
    - The project owner can modify their project.

    Args:
        executing_user_id: The integer user ID performing the action.
        prjct_id: The integer project ID.

    Returns:
        True if the user can modify the project, otherwise False.

    Raises:
        TypeError: If inputs are not integers.
        ValueError: If the project_id or user_id are unknown.
    """
    if not isinstance(executing_user_id, int):
        raise TypeError("executing_user_id must be an integer")
    if not isinstance(prjct_id, int):
        raise TypeError("prjct_id must be an integer")

    try:
        project = PROJECTS[prjct_id]
    except KeyError:
        raise ValueError(f"Unknown project_id: {prjct_id}")

    role = check_user_role(executing_user_id)
    if role == "admin":
        return True

    return project["owner_id"] == executing_user_id


def control_project_permissions(
    executing_user_id: int,
    prjct_id: int,
    act_type: str,
    tgt_user_id: int
) -> bool:
    """
    Manage project membership by allowing admins to manage any project and
    project owners to manage their own projects.

    Supported actions (act_type):
    - "add": add tgt_user_id to the project's members list
    - "remove": remove tgt_user_id from the project's members list

    Rules:
    - Admins can manage any project.
    - Only the project owner can manage their project if not admin.
    - Cannot remove the project owner from the members list.

    Returns:
        True if the operation succeeds (membership list changed accordingly), else False.
    """
    # Basic type checks to avoid raising from helper functions
    if not isinstance(executing_user_id, int):
        return False
    if not isinstance(prjct_id, int):
        return False
    if not isinstance(tgt_user_id, int):
        return False
    if not isinstance(act_type, str):
        return False

    action = act_type.strip().lower()
    if action not in ("add", "remove"):
        return False

    # Validate project exists
    project = PROJECTS.get(prjct_id)
    if project is None:
        return False

    # Check permission (admin or owner)
    try:
        if not can_modify_project(executing_user_id, prjct_id):
            return False
    except (TypeError, ValueError):
        return False

    # Validate target user exists
    if tgt_user_id not in USERS:
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False  # malformed project structure

    if action == "add":
        if tgt_user_id in members:
            return False
        members.append(tgt_user_id)
        return True

    # action == "remove"
    if tgt_user_id == project.get("owner_id"):
        return False  # cannot remove the owner from their own project
    if tgt_user_id not in members:
        return False
    members.remove(tgt_user_id)
    return True
