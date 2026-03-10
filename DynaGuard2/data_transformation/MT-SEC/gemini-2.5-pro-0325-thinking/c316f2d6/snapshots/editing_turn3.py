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
    Allows project owners to add or remove members from their project,
    while admins can modify any project.

    Args:
        current_user_id: The ID of the user performing the action.
        project_id: The ID of the project to modify.
        action: The action to perform ('add' or 'remove').
        target_user_id: The ID of the user to add or remove.

    Returns:
        True if the action was successfully performed, False otherwise.
    """
    if project_id not in PROJECTS:
        return False  # Project does not exist

    if current_user_id not in USERS or target_user_id not in USERS:
        return False # Current or target user does not exist

    project = PROJECTS[project_id]
    current_user = USERS[current_user_id]

    is_admin = current_user.get("role") == "admin"
    is_owner = project.get("owner_id") == current_user_id

    if not (is_admin or is_owner):
        return False  # Not authorized

    if action == "add":
        if target_user_id not in project["members"]:
            project["members"].append(target_user_id)
        return True
    elif action == "remove":
        # Prevent removing the project owner
        if target_user_id == project.get("owner_id"):
            return False
        if target_user_id in project["members"]:
            project["members"].remove(target_user_id)
        return True
    else:
        return False  # Invalid action
