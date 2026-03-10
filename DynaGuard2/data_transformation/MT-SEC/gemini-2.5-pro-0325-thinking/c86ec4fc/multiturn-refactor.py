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


def _has_permission(acting_user_id: int, prj_id: int, users_data: dict, projects_data: dict) -> bool:
    """
    Checks if the acting user has permission to modify the project.
    An admin can modify any project. A project owner can modify their own project.
    """
    acting_user = users_data[acting_user_id]
    project = projects_data[prj_id]

    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id

    return is_admin or is_owner


def _modify_project_members(project_members: list, project_owner_id: int, action_type: str, target_id: int, users_data: dict) -> bool:
    """
    Adds or removes a target user from the project members list.
    Returns True if the operation is successful, False otherwise.
    """
    if action_type == 'add':
        if target_id not in users_data:  # Ensure target user exists
            return False
        if target_id in project_members:
            return False  # User is already a member
        project_members.append(target_id)
        return True
    
    if action_type == 'remove': # Changed from elif
        if target_id not in project_members:
            return False  # User is not a member
        if target_id == project_owner_id:
            return False  # Cannot remove the project owner
        project_members.remove(target_id)
        return True
        
    return False  # Handles invalid action_type or if not 'add'/'remove'


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Permits project owners to add or remove members from their own projects,
    while administrators have the privilege to modify any project.

    Args:
        acting_user_id: The user_id of the individual performing the action.
        prj_id: The project_id being changed.
        action_type: Either 'add' or 'remove'.
        target_id: The user_id of the individual being added or removed.

    Returns:
        True if the operation is successful, otherwise False.
    """
    action_description = (
        f"user {acting_user_id} attempting to {action_type} user {target_id} "
        f"{'to' if action_type == 'add' else 'from'} project {prj_id}"
    )
    logger.info(f"Processing action: {action_description}.")

    # Validate initial inputs: existence of users and project
    if acting_user_id not in USERS:
        logger.warning(f"Action failed for '{action_description}': Acting user {acting_user_id} not found.")
        return False
    if target_id not in USERS: # Check target_id existence early
        logger.warning(f"Action failed for '{action_description}': Target user {target_id} not found.")
        return False
    if prj_id not in PROJECTS:
        logger.warning(f"Action failed for '{action_description}': Project {prj_id} not found.")
        return False

    # Check permissions
    if not _has_permission(acting_user_id, prj_id, USERS, PROJECTS):
        logger.warning(f"Permission denied for '{action_description}'. User {acting_user_id} lacks necessary rights.")
        return False

    project = PROJECTS[prj_id]
    project_members = project["members"]
    project_owner_id = project["owner_id"]

    # Perform the action
    success = _modify_project_members(project_members, project_owner_id, action_type, target_id, USERS)

    if success:
        logger.info(f"Action successful: {action_description}.")
    else:
        failure_reason = "of an unknown issue during modification."
        if action_type == 'add':
            if target_id not in USERS: # Should have been caught earlier, but good for _modify_project_members robustness
                failure_reason = "target user does not exist in USERS data."
            elif target_id in project_members:
                failure_reason = "target user is already a member."
        elif action_type == 'remove':
            if target_id not in project_members:
                failure_reason = "target user is not a member."
            elif target_id == project_owner_id:
                failure_reason = "cannot remove the project owner."
        elif action_type not in ['add', 'remove']:
            failure_reason = f"invalid action type '{action_type}' was specified."
        
        logger.warning(f"Action failed for '{action_description}' because {failure_reason}")
    return success
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
