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
    """
    if project_id not in PROJECTS:
        return False  # Project does not exist

    if current_user_id not in USERS:
        return False  # Current user does not exist

    project = PROJECTS[project_id]
    current_user = USERS[current_user_id]

    is_admin = current_user["role"] == "admin"
    is_owner = project["owner_id"] == current_user_id

    if not (is_admin or is_owner):
        return False  # Not authorized

    if action == "add":
        if target_user_id not in USERS:
            return False  # Target user does not exist, cannot add

        if target_user_id not in project["members"]:
            project["members"].append(target_user_id)
        return True
    elif action == "remove":
        if target_user_id == project["owner_id"]:
            return False  # Cannot remove the project owner

        if target_user_id in project["members"]:
            project["members"].remove(target_user_id)
        return True
    else:
        return False  # Invalid action
