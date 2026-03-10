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


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Permits project owners to add or remove members from their own projects,
    while administrators can modify any project.

    Args:
        acting_user_id: The user_id performing the action.
        prj_id: The project being changed.
        action_type: Either 'add' or 'remove'.
        target_id: The user_id being added or removed.

    Returns:
        True if the operation is successful (membership changed), otherwise False.
    """
    # Validate basic inputs
    if prj_id not in PROJECTS:
        return False
    if acting_user_id not in USERS:
        return False
    if target_id not in USERS:
        return False
    if action_type not in ("add", "remove"):
        return False

    project = PROJECTS[prj_id]
    members = project.get("members", [])
    owner_id = project.get("owner_id")

    # Authorization: admin or project owner
    acting_role = USERS[acting_user_id].get("role")
    is_admin = acting_role == "admin"
    is_owner = acting_user_id == owner_id
    if not (is_admin or is_owner):
        return False

    # Perform action
    if action_type == "add":
        if target_id in members:
            return False
        members.append(target_id)
        return True

    # action_type == "remove"
    if target_id not in members:
        return False
    members.remove(target_id)
    return True
