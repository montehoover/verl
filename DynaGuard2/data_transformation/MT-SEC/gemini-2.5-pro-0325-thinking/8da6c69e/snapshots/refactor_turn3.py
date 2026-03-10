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

def handle_project_membership(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """
    Manages project membership. Owners can add/remove members from their projects.
    Admins can manage any project.
    """
    success = False
    if not _validate_inputs(acting_user_id, prjt_id, member_action, target_member_id):
        logging.warning(
            f"Validation failed for handle_project_membership: "
            f"acting_user_id={acting_user_id}, prjt_id={prjt_id}, "
            f"member_action='{member_action}', target_member_id={target_member_id}"
        )
        return False

    if not _is_authorized(acting_user_id, prjt_id):
        logging.warning(
            f"Authorization failed for handle_project_membership: "
            f"acting_user_id={acting_user_id} is not authorized for prjt_id={prjt_id}."
        )
        return False

    if member_action == 'add':
        success = _add_member_to_project(prjt_id, target_member_id)
    elif member_action == 'remove':
        success = _remove_member_from_project(prjt_id, target_member_id)
    else:
        # This case should ideally not be reached if _validate_inputs is correct
        logging.error(f"Invalid member_action '{member_action}' encountered in handle_project_membership.")
        return False

    log_message = (
        f"Action: {member_action}, Project ID: {prjt_id}, "
        f"Target Member ID: {target_member_id}, Acting User ID: {acting_user_id}, "
        f"Success: {success}"
    )
    if success:
        logging.info(log_message)
    else:
        logging.error(log_message)
        
    return success


def _validate_inputs(acting_user_id: int, prjt_id: int, member_action: str, target_member_id: int) -> bool:
    """Validates the inputs for handle_project_membership."""
    if acting_user_id not in USERS or target_member_id not in USERS:
        return False  # Invalid user ID
    if prjt_id not in PROJECTS:
        return False  # Invalid project ID
    if member_action not in ['add', 'remove']:
        return False  # Invalid action
    return True


def _is_authorized(acting_user_id: int, prjt_id: int) -> bool:
    """Checks if the acting user is authorized to manage the project."""
    acting_user = USERS[acting_user_id]
    project = PROJECTS[prjt_id]

    is_admin = acting_user["role"] == "admin"
    is_owner = project["owner_id"] == acting_user_id

    return is_admin or is_owner


def _add_member_to_project(prjt_id: int, target_member_id: int) -> bool:
    """Adds a member to the specified project."""
    project = PROJECTS[prjt_id]
    if target_member_id not in project["members"]:
        project["members"].append(target_member_id)
        return True
    return False  # Member already exists or other error


def _remove_member_from_project(prjt_id: int, target_member_id: int) -> bool:
    """Removes a member from the specified project."""
    project = PROJECTS[prjt_id]
    # Prevent owner from being removed
    if target_member_id == project["owner_id"]:
        return False

    if target_member_id in project["members"]:
        project["members"].remove(target_member_id)
        return True
    return False  # Member not found or other error
import logging

# Basic logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
