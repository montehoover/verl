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

def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows project owners to add or remove team members from their own projects,
    while admins can manage any project.

    Args:
        executor_user_id: The ID of the user performing the task.
        proj_id: The ID of the project.
        operation: 'add' or 'remove'.
        target_member_id: The ID of the user to be added or removed.

    Returns:
        True if the modification is successful, False otherwise.
    """
    if executor_user_id not in USERS or proj_id not in PROJECTS:
        return False  # Executor or project does not exist

    executor_user = USERS[executor_user_id]
    project = PROJECTS[proj_id]

    is_admin = executor_user.get("role") == "admin"
    is_owner = project.get("owner_id") == executor_user_id

    if not (is_admin or is_owner):
        return False  # Not authorized

    if target_member_id not in USERS:
        return False # Target member does not exist

    project_members = project.get("members", [])

    if operation == "add":
        if target_member_id not in project_members:
            project_members.append(target_member_id)
            PROJECTS[proj_id]["members"] = project_members
            return True
        else:
            return False  # Member already exists
    elif operation == "remove":
        if target_member_id in project_members:
            # Prevent owner from removing themselves if they are the only member
            if target_member_id == project.get("owner_id") and len(project_members) == 1:
                return False
            project_members.remove(target_member_id)
            PROJECTS[proj_id]["members"] = project_members
            return True
        else:
            return False  # Member not found
    else:
        return False  # Invalid operation
