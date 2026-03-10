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
    Permits project owners to add/remove members from their own projects.
    Administrators may modify any project.

    Returns True on successful modification, False otherwise.
    """
    # Validate existence of acting user and project
    if acting_user_id not in USERS or prj_id not in PROJECTS:
        return False

    action_type = action_type.lower().strip()
    if action_type not in ("add", "remove"):
        return False

    project = PROJECTS[prj_id]
    owner_id = project.get("owner_id")
    members = project.get("members")

    if not isinstance(members, list):
        return False

    acting_user_role = USERS[acting_user_id].get("role")

    is_admin = acting_user_role == "admin"
    is_owner = acting_user_id == owner_id

    # Authorization check
    if not (is_admin or is_owner):
        return False

    # Validate target existence
    if target_id not in USERS:
        return False

    if action_type == "add":
        if target_id in members:
            return False
        members.append(target_id)
        return True

    # action_type == "remove"
    # Do not allow removing the owner from the project
    if target_id == owner_id:
        return False

    if target_id not in members:
        return False

    # Remove the target user
    try:
        members.remove(target_id)
        return True
    except ValueError:
        return False
