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
logger.addHandler(logging.NullHandler())


def has_project_management_permission(active_user: dict, project: dict) -> bool:
    """
    Pure function: Determine if the active_user can manage the given project.
    Admins can manage any project; owners can manage their own project.
    """
    if not active_user or not project:
        return False
    return (
        active_user.get("role") == "admin"
        or project.get("owner_id") == active_user.get("user_id")
    )


def compute_members_update(action_type: str, members, member_user_id: int):
    """
    Pure function: Compute the new members list based on the requested action.
    Returns (success: bool, new_members: list). Does not mutate the input list.
    """
    if not isinstance(members, list):
        return False, members

    if action_type == "add":
        return (False, members) if member_user_id in members else (True, members + [member_user_id])

    if action_type == "remove":
        if member_user_id not in members:
            return False, members
        return True, [m for m in members if m != member_user_id]

    # Unknown action (should be validated by caller)
    return False, members


def project_access_control(active_user_id: int, prj_id: int, action_type: str, member_user_id: int) -> bool:
    """
    Allow project owners to add/remove members in their own projects,
    and admins to manage any project without restriction.

    Returns True if the operation was performed, False otherwise.
    """
    logger.debug(
        "project_access_control called with active_user_id=%s, prj_id=%s, action_type=%s, member_user_id=%s",
        active_user_id, prj_id, action_type, member_user_id
    )

    if action_type not in ("add", "remove"):
        logger.error("Invalid action_type '%s'. Expected 'add' or 'remove'.", action_type)
        return False

    project = PROJECTS.get(prj_id)
    if project is None:
        logger.error("Project %s not found.", prj_id)
        return False

    active_user = USERS.get(active_user_id)
    if active_user is None:
        logger.error("Active user %s not found.", active_user_id)
        return False

    target_user = USERS.get(member_user_id)
    if target_user is None:
        logger.error("Target user %s not found.", member_user_id)
        return False

    if not has_project_management_permission(active_user, project):
        logger.warning(
            "Permission denied: user %s (role=%s) cannot %s member %s in project %s (owner=%s).",
            active_user_id, active_user.get("role"), action_type, member_user_id, prj_id, project.get("owner_id")
        )
        return False

    members = project.setdefault("members", [])
    if not isinstance(members, list):
        logger.error("Invalid 'members' type for project %s: %s", prj_id, type(members).__name__)
        return False

    success, new_members = compute_members_update(action_type, members, member_user_id)
    if not success:
        reason = (
            "already a member"
            if action_type == "add" and member_user_id in members
            else "not a member"
            if action_type == "remove" and member_user_id not in members
            else "update rejected"
        )
        logger.info(
            "No change for project %s: user %s attempted to %s member %s (%s).",
            prj_id, active_user_id, action_type, member_user_id, reason
        )
        return False

    project["members"] = new_members
    logger.info(
        "Success: user %s %s member %s in project %s. New members: %s",
        active_user_id, "added" if action_type == "add" else "removed", member_user_id, prj_id, new_members
    )
    return True
