# This file implements update_project_members based on the provided setup.
# Extracted pure helper functions for permission checking and member modification.
# Added logging for successful membership modifications and simplified conditionals.

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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


def normalize_modification(modification: str):
    """
    Normalize the modification input to 'add' or 'remove'.
    Returns 'add' or 'remove' if valid; otherwise None.
    """
    if not isinstance(modification, str):
        return None
    mod = modification.strip().lower()
    return mod if mod in ("add", "remove") else None


def is_authorized(acting_user_role: str, project_owner_id: int, acting_user_id: int) -> bool:
    """
    Determine if the acting user is authorized to modify the project.
    Admins can manage any project.
    Project owners can manage only their own projects.
    """
    return acting_user_role == "admin" or project_owner_id == acting_user_id


def compute_membership_change(current_members, mod: str, user_to_update_id: int):
    """
    Pure function that computes the new membership list based on the operation.
    Returns (success: bool, new_members: list)
    - For 'add': succeeds only if the user is not already a member.
    - For 'remove': succeeds only if the user is currently a member.
    """
    if not isinstance(current_members, list):
        return False, current_members

    new_members = list(current_members)
    is_member = user_to_update_id in new_members

    if mod == "add":
        if is_member:
            return False, current_members
        new_members.append(user_to_update_id)
        return True, new_members

    if mod == "remove":
        if not is_member:
            return False, current_members
        new_members.remove(user_to_update_id)
        return True, new_members

    return False, current_members


def update_project_members(
    acting_user_id: int,
    project_identifier: int,
    modification: str,
    user_to_update_id: int
) -> bool:
    """
    Update project membership by adding or removing a user.

    Permissions:
    - Admins can manage any project.
    - Project owners can manage only their own projects.

    Args:
        acting_user_id: The ID of the user performing the action.
        project_identifier: The ID of the project to modify.
        modification: 'add' or 'remove'.
        user_to_update_id: The ID of the user to add/remove.

    Returns:
        True if the modification is successful, otherwise False.
    """
    project = PROJECTS.get(project_identifier)
    if not isinstance(project, dict):
        return False

    if "owner_id" not in project:
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    acting_user = USERS.get(acting_user_id)
    if not isinstance(acting_user, dict) or "role" not in acting_user:
        return False

    if user_to_update_id not in USERS:
        return False

    mod = normalize_modification(modification)
    if mod is None:
        return False

    if not is_authorized(acting_user.get("role"), project.get("owner_id"), acting_user_id):
        return False

    # Compute membership change (pure), then apply if successful
    before_members = list(members)
    success, new_members = compute_membership_change(members, mod, user_to_update_id)
    if success:
        project["members"] = new_members
        logger.info(
            "Project membership updated: actor=%s action=%s target=%s project=%s before=%s after=%s",
            acting_user_id,
            mod,
            user_to_update_id,
            project_identifier,
            before_members,
            new_members,
        )
    return success
