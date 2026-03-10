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


def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Manage membership for a project.

    Permissions:
    - Admins (role == 'admin') can add/remove members for any project.
    - Project owners can add/remove members only for their own project.

    Args:
        current_user_id: The user_id of the actor invoking the function.
        project_id: The project being modified.
        action: 'add' or 'remove' (case-insensitive).
        target_user_id: The user_id to be added or removed.

    Returns:
        True if the action was successfully performed (i.e., membership changed), else False.
    """
    # Validate existence of users and project
    if project_id not in PROJECTS:
        return False
    if current_user_id not in USERS or target_user_id not in USERS:
        return False

    project = PROJECTS[project_id]
    current_role = USERS[current_user_id].get("role")

    # Authorization: admin or project owner
    is_admin = current_role == "admin"
    is_owner = project.get("owner_id") == current_user_id
    if not (is_admin or is_owner):
        return False

    # Validate action
    if not isinstance(action, str):
        return False
    action = action.lower()
    if action not in ("add", "remove"):
        return False

    members = project.setdefault("members", [])

    if action == "add":
        if target_user_id in members:
            return False
        members.append(target_user_id)
        return True

    # action == "remove"
    if target_user_id not in members:
        return False
    members.remove(target_user_id)
    return True
