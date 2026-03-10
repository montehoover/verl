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


def _has_permission(current_user_id: int, project_id: int, users_data: dict, projects_data: dict) -> bool:
    """Checks if the current user has permission to modify the project."""
    current_user = users_data[current_user_id]
    project = projects_data[project_id]
    is_admin = current_user["role"] == "admin"
    is_owner = project["owner_id"] == current_user_id
    return is_admin or is_owner


def _modify_project_members(project_id: int, action: str, target_user_id: int, projects_data: dict) -> bool:
    """Adds or removes a member from a project's member list."""
    project = projects_data[project_id]
    if action == 'add':
        if target_user_id in project["members"]:
            return False  # User is already a member
        project["members"].append(target_user_id)
        return True
    elif action == 'remove':
        if target_user_id == project["owner_id"]:
            return False  # Owner cannot be removed
        if target_user_id not in project["members"]:
            return False  # User is not a member
        project["members"].remove(target_user_id)
        return True
    return False  # Invalid action


def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Manages access to a project by adding or removing members.

    Args:
        current_user_id: The ID of the user performing the action.
        project_id: The ID of the project to modify.
        action: The action to perform ('add' or 'remove').
        target_user_id: The ID of the user to add or remove.

    Returns:
        True if the action was successful, False otherwise.
    """
    log_prefix = f"User {current_user_id} attempting to {action} user {target_user_id} for project {project_id}:"

    if current_user_id not in USERS or target_user_id not in USERS:
        logging.warning(f"{log_prefix} Failed - Actor or target user does not exist.")
        return False

    if project_id not in PROJECTS:
        logging.warning(f"{log_prefix} Failed - Project does not exist.")
        return False

    if not _has_permission(current_user_id, project_id, USERS, PROJECTS):
        logging.warning(f"{log_prefix} Failed - Permission denied.")
        return False

    success = _modify_project_members(project_id, action, target_user_id, PROJECTS)

    if success:
        logging.info(f"{log_prefix} Succeeded.")
    else:
        logging.warning(f"{log_prefix} Failed - Action '{action}' could not be completed (e.g., user already member/not member, or trying to remove owner).")
    return success
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
