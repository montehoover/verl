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
    Manages team access for projects.

    Args:
        executor_user_id: The user_id of the individual performing the action.
        proj_id: The project_id being changed.
        operation: Either 'add' or 'remove'.
        target_member_id: The user_id of the individual being added or removed.

    Returns:
        True if the operation is successful, False otherwise.
    """
    # Validate existence of executor, project, and target member
    if executor_user_id not in USERS:
        return False  # Executor user does not exist
    if proj_id not in PROJECTS:
        return False  # Project does not exist
    if target_member_id not in USERS:
        return False  # Target member user does not exist

    executor = USERS[executor_user_id]
    project = PROJECTS[proj_id]

    # Check permissions: Admin can modify any project, owner can modify their own project
    is_admin = executor["role"] == "admin"
    is_owner = project["owner_id"] == executor_user_id

    if not (is_admin or is_owner):
        return False  # Insufficient permissions

    # Perform the operation
    if operation == "add":
        # Add member if not already present
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True  # Successful, even if member was already present (idempotent)
    elif operation == "remove":
        # Cannot remove the project owner
        if target_member_id == project["owner_id"]:
            return False

        # Remove member if present
        if target_member_id in project["members"]:
            project["members"].remove(target_member_id)
            return True
        else:
            return False  # Member not found in project, cannot remove
    else:
        return False  # Invalid operation
