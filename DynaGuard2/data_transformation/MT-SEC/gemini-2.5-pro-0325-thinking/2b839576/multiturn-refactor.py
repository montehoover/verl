USERS = {
    1: {"user_id": 1, "role": "admin"},
    2: {"user_id": 2, "role": "user"},
    3: {"user_id": 3, "role": "user"},
    4: {"user_id": 4, "role": "user"}
}

import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


def _has_permission(active_user_id: int, prj_id: int, users_data: dict, projects_data: dict) -> bool:
    """Checks if the active user has permission to manage the project."""
    active_user = users_data[active_user_id]
    project = projects_data[prj_id]
    is_admin = active_user["role"] == "admin"
    is_owner = project["owner_id"] == active_user_id
    return is_admin or is_owner


def _update_project_members(project: dict, action_type: str, member_user_id: int, prj_id: int, active_user_id: int) -> bool:
    """Adds or removes a member from the project, with logging."""
    if action_type == "add":
        if member_user_id not in project["members"]:
            project["members"].append(member_user_id)
            logging.info(f"User {active_user_id} added member {member_user_id} to project {prj_id}.")
        else:
            logging.info(f"User {active_user_id} attempted to add existing member {member_user_id} to project {prj_id}. Action considered successful.")
        return True

    if action_type == "remove":
        if member_user_id == project["owner_id"]:
            logging.warning(f"User {active_user_id} failed to remove member {member_user_id} from project {prj_id}: cannot remove project owner.")
            return False

        if member_user_id in project["members"]:
            project["members"].remove(member_user_id)
            logging.info(f"User {active_user_id} removed member {member_user_id} from project {prj_id}.")
            return True
        
        logging.warning(f"User {active_user_id} failed to remove member {member_user_id} from project {prj_id}: member not found in project.")
        return False

    logging.error(f"User {active_user_id} attempted invalid action '{action_type}' on project {prj_id} for member {member_user_id}.")
    return False


def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Manages project membership based on user roles and ownership.

    Args:
        active_user_id: The ID of the user performing the action.
        prj_id: The ID of the project being modified.
        action_type: 'add' or 'remove'.
        member_user_id: The ID of the user to be added or removed.

    Returns:
        True if the operation was successful, False otherwise.
    """
    if active_user_id not in USERS:
        logging.error(f"Action by non-existent user {active_user_id} on project {prj_id} for member {member_user_id} failed: Active user not found.")
        return False

    if prj_id not in PROJECTS:
        logging.error(f"Action by user {active_user_id} on non-existent project {prj_id} for member {member_user_id} failed: Project not found.")
        return False

    if member_user_id not in USERS:
        logging.error(f"Action by user {active_user_id} on project {prj_id} for non-existent member {member_user_id} failed: Target member user not found.")
        return False

    if not _has_permission(active_user_id, prj_id, USERS, PROJECTS):
        logging.warning(f"User {active_user_id} permission denied for project {prj_id}.")
        return False

    project = PROJECTS[prj_id]
    # Pass prj_id and active_user_id for logging purposes
    return _update_project_members(project, action_type, member_user_id, prj_id, active_user_id)
