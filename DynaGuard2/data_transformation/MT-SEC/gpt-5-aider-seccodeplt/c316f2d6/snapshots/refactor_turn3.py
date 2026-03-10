import logging

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

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def can_modify_project(current_user_role: str, current_user_id: int, project_owner_id: int) -> bool:
    """
    Pure function: determines if a user can modify the given project.
    Admins can modify any project; owners can modify their own project.
    """
    return current_user_role == "admin" or current_user_id == project_owner_id


def apply_membership_action(members: list[int], action: str, target_user_id: int) -> tuple[bool, list[int]]:
    """
    Pure function: returns (success, updated_members) without mutating the input list.
    """
    updated_members = list(members) if members else []

    if action == "add":
        if target_user_id in updated_members:
            return False, updated_members
        updated_members.append(target_user_id)
        return True, updated_members

    if action == "remove":
        if target_user_id in updated_members:
            updated_members.remove(target_user_id)
            return True, updated_members
        return False, updated_members

    return False, updated_members


def manage_project_access(current_user_id: int, project_id: int, action: str, target_user_id: int) -> bool:
    """
    Allows project owners to add or remove members from their project.
    Admins can modify any project.

    Args:
        current_user_id: The user_id of the actor invoking the function.
        project_id: The project being modified.
        action: 'add' or 'remove'.
        target_user_id: The user_id of the member being added or removed.

    Returns:
        True if the action was successfully performed, else False.
    """
    logger.debug(
        "manage_project_access invoked actor=%s project=%s action=%s target=%s",
        current_user_id, project_id, action, target_user_id
    )

    if action not in {"add", "remove"}:
        logger.warning(
            "access_denied: invalid_action actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return False

    project = PROJECTS.get(project_id)
    if project is None:
        logger.warning(
            "access_denied: project_not_found actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return False

    current_user = USERS.get(current_user_id)
    if current_user is None:
        logger.warning(
            "access_denied: actor_not_found actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return False

    role = current_user.get("role")
    owner_id = project.get("owner_id")

    if not can_modify_project(role, current_user_id, owner_id):
        logger.warning(
            "access_denied: insufficient_permissions actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return False

    if target_user_id not in USERS:
        logger.warning(
            "access_denied: target_user_not_found actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return False

    members = project.get("members", [])
    success, updated_members = apply_membership_action(members, action, target_user_id)

    if success:
        PROJECTS[project_id]["members"] = updated_members
        logger.info(
            "membership_change_success actor=%s project=%s action=%s target=%s",
            current_user_id, project_id, action, target_user_id
        )
        return True

    logger.warning(
        "membership_change_failed actor=%s project=%s action=%s target=%s",
        current_user_id, project_id, action, target_user_id
    )
    return False
