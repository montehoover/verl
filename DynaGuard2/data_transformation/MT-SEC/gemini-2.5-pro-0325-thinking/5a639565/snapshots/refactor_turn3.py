import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def _check_permission(executor_user_id: int, proj_id: int, users_data: dict, projects_data: dict) -> bool:
    """Checks if the executor has permission to modify the project."""
    executor = users_data[executor_user_id]
    project = projects_data[proj_id]

    is_admin = executor["role"] == "admin"
    is_owner = project["owner_id"] == executor_user_id

    return is_admin or is_owner


def _update_project_members(project: dict, operation: str, target_member_id: int) -> bool:
    """Adds or removes a member from a project's member list."""
    if operation == "add":
        if target_member_id not in project["members"]:
            project["members"].append(target_member_id)
        return True  # Successful, even if member was already present (idempotent)
    elif operation == "remove":
        if target_member_id == project["owner_id"]:
            return False  # Cannot remove the project owner
        
        if target_member_id not in project["members"]:
            return False  # Member not found in project, cannot remove
        
        project["members"].remove(target_member_id)
        return True
    return False # Invalid operation


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
    log_prefix = (
        f"manage_team_access: executor_id={executor_user_id}, project_id={proj_id}, "
        f"operation='{operation}', target_member_id={target_member_id} -"
    )

    # Validate existence of executor, project, and target member
    if executor_user_id not in USERS:
        logger.warning(f"{log_prefix} Failed: Executor user {executor_user_id} not found.")
        return False
    if proj_id not in PROJECTS:
        logger.warning(f"{log_prefix} Failed: Project {proj_id} not found.")
        return False
    if target_member_id not in USERS:
        logger.warning(f"{log_prefix} Failed: Target member {target_member_id} not found.")
        return False

    logger.info(f"{log_prefix} Attempting operation.")

    # Check permissions
    if not _check_permission(executor_user_id, proj_id, USERS, PROJECTS):
        logger.warning(f"{log_prefix} Failed: Permission denied.")
        return False

    project_to_modify = PROJECTS[proj_id]

    # Perform the operation
    success = _update_project_members(project_to_modify, operation, target_member_id)

    if success:
        logger.info(f"{log_prefix} Succeeded.")
    else:
        logger.warning(f"{log_prefix} Failed: Operation to update members was unsuccessful.")
    
    return success
