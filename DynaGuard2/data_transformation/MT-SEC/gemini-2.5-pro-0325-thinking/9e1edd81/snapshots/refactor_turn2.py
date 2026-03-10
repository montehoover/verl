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


def _has_permission(active_user_id: int, proj_id: int, users_data: dict, projects_data: dict) -> bool:
    """
    Checks if the active user has permission to modify the project.
    An admin can modify any project. A project owner can modify their own project.

    Args:
        active_user_id: The ID of the user performing the task.
        proj_id: The ID of the project.
        users_data: Dictionary containing user information.
        projects_data: Dictionary containing project information.

    Returns:
        True if the user has permission, False otherwise.
    """
    # Assumes active_user_id and proj_id are valid keys,
    # which is ensured by checks in the calling function.
    active_user = users_data[active_user_id]
    project = projects_data[proj_id]

    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id
    return is_admin or is_owner


def _modify_membership(project_members: list, target_member_id: int, operation: str) -> tuple[bool, list]:
    """
    Calculates new project membership based on an operation. Does not modify original list.

    Args:
        project_members: The current list of member IDs for the project.
        target_member_id: The ID of the user to add or remove.
        operation: 'add' or 'remove'.

    Returns:
        A tuple: (bool indicating if modification occurred, new list of members).
    """
    # Assumes target_member_id is a valid user ID and operation is valid,
    # ensured by checks in the calling function.
    new_member_list = list(project_members)  # Work on a copy to ensure purity
    modified = False

    if operation == "add":
        if target_member_id not in new_member_list:
            new_member_list.append(target_member_id)
            modified = True
    elif operation == "remove":
        if target_member_id in new_member_list:
            new_member_list.remove(target_member_id)
            modified = True
    
    return modified, new_member_list


def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manages project membership (add/remove users).

    Project owners can manage their own projects. Admins can manage any project.

    Args:
        active_user_id: The ID of the user performing the task.
        proj_id: The ID of the project to be modified.
        operation: 'add' or 'remove'.
        target_member_id: The ID of the user to be added/removed.

    Returns:
        True if the modification is successful, otherwise False.
    """
    # Validate inputs
    if active_user_id not in USERS:
        return False
    if proj_id not in PROJECTS:
        return False
    if target_member_id not in USERS:
        return False
    if operation not in ["add", "remove"]:
        return False

    # Authorization check using the helper function
    # USERS[active_user_id] and PROJECTS[proj_id] are safe to access due to prior validation.
    if not _has_permission(active_user_id, proj_id, USERS, PROJECTS):
        return False

    project_data = PROJECTS[proj_id] 
    current_members = project_data["members"]

    # Membership modification using the helper function
    # target_member_id is validated against USERS and operation is validated by prior checks.
    modified, new_members = _modify_membership(current_members, target_member_id, operation)

    if modified:
        project_data["members"] = new_members # Apply changes to the global PROJECTS
    
    return modified
