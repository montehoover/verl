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

def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Allows project owners to add or remove team members from their own projects,
    while admins can manage any project.

    Args:
        active_user_id: The ID of the user performing the task.
        prj_id: The ID of the project.
        action_type: 'add' or 'remove'.
        member_user_id: The ID of the user to be added or removed.

    Returns:
        True if the modification is successful, False otherwise.
    """
    if prj_id not in PROJECTS:
        return False  # Project does not exist

    if active_user_id not in USERS or member_user_id not in USERS:
        return False # Active user or member user does not exist

    active_user = USERS[active_user_id]
    project = PROJECTS[prj_id]

    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id

    if not (is_admin or is_owner):
        return False  # User is not authorized to manage this project

    if action_type == "add":
        if member_user_id not in project["members"]:
            project["members"].append(member_user_id)
            return True
        return False # User already a member or no change made
    elif action_type == "remove":
        if member_user_id in project["members"]:
            # Optional: Prevent owner from removing themselves if that's a desired rule.
            # if member_user_id == project["owner_id"] and member_user_id == active_user_id and not is_admin:
            #     return False # Owner cannot remove themselves unless an admin is doing it
            project["members"].remove(member_user_id)
            return True
        return False # User not a member or no change made
    else:
        return False  # Invalid action type
