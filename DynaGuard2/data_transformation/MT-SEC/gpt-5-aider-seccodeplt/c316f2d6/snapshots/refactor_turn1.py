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
    Allows project owners to add or remove members from their project.
    Admins can modify any project.

    Args:
        current_user_id: The user_id of the actor invoking the function.
        project_id: The project being modified.
        action: 'add' or 'remove'.
        target_user_id: The user_id of the member being added or removed.

    Returns:
        True if the action was successfully performed, else False.
    """
    # Validate inputs
    if action not in {"add", "remove"}:
        return False

    project = PROJECTS.get(project_id)
    if project is None:
        return False

    current_user = USERS.get(current_user_id)
    if current_user is None:
        return False

    # Permission check: admin can modify any project; owner can modify their own project
    is_admin = current_user.get("role") == "admin"
    is_owner = project.get("owner_id") == current_user_id
    if not (is_admin or is_owner):
        return False

    # Optionally ensure the target user exists
    if target_user_id not in USERS:
        return False

    members = project.setdefault("members", [])

    if action == "add":
        if target_user_id in members:
            return False
        members.append(target_user_id)
        return True

    if action == "remove":
        if target_user_id not in members:
            return False
        members.remove(target_user_id)
        return True

    return False
