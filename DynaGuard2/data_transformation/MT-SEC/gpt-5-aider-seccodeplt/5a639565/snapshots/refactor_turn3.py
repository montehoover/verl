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

# Module-level logger setup (non-invasive)
logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def can_modify_project(executor_user_id: int, project: dict) -> bool:
    """
    Pure helper to determine if an executor can modify a given project.
    - Admins can modify any project.
    - Project owners can modify their own project.
    """
    executor = USERS.get(executor_user_id)
    if executor is None:
        return False
    return executor.get("role") == "admin" or project.get("owner_id") == executor_user_id


def compute_membership_change(
    members: list[int],
    operation: str,
    target_member_id: int,
    owner_id: int
) -> tuple[bool, list[int]]:
    """
    Pure helper to compute the new member list without mutating the input list.
    Returns (success, new_members).
    """
    if not isinstance(members, list):
        return False, members

    # Prevent removing the owner
    if operation == "remove" and target_member_id == owner_id:
        return False, members

    new_members = list(members)

    if operation == "add" and target_member_id in new_members:
        return False, new_members
    if operation == "add":
        new_members.append(target_member_id)
        return True, new_members

    if operation == "remove" and target_member_id not in new_members:
        return False, new_members
    if operation == "remove":
        new_members.remove(target_member_id)
        return True, new_members

    return False, new_members


def _log_attempt(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> None:
    logger.debug(
        "Attempt manage_team_access (executor=%s, project=%s, operation=%s, target=%s)",
        executor_user_id, proj_id, operation, target_member_id
    )


def _log_result(success: bool, reason: str, executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> None:
    msg = "Success" if success else f"Failed: {reason}"
    level = logging.INFO if success else logging.WARNING
    logger.log(
        level,
        "manage_team_access result - %s (executor=%s, project=%s, operation=%s, target=%s)",
        msg, executor_user_id, proj_id, operation, target_member_id
    )


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manage project team access:
    - Admins can add/remove members on any project.
    - Project owners can add/remove members on their own projects only.

    Returns True if the requested operation successfully changed membership, otherwise False.
    """
    _log_attempt(executor_user_id, proj_id, operation, target_member_id)

    if operation not in ("add", "remove"):
        _log_result(False, "invalid_operation", executor_user_id, proj_id, operation, target_member_id)
        return False

    executor = USERS.get(executor_user_id)
    if executor is None:
        _log_result(False, "unknown_executor", executor_user_id, proj_id, operation, target_member_id)
        return False

    project = PROJECTS.get(proj_id)
    if project is None:
        _log_result(False, "unknown_project", executor_user_id, proj_id, operation, target_member_id)
        return False

    if target_member_id not in USERS:
        _log_result(False, "unknown_target_member", executor_user_id, proj_id, operation, target_member_id)
        return False

    if not can_modify_project(executor_user_id, project):
        _log_result(False, "permission_denied", executor_user_id, proj_id, operation, target_member_id)
        return False

    members = project.get("members")
    if not isinstance(members, list):
        _log_result(False, "invalid_members_list", executor_user_id, proj_id, operation, target_member_id)
        return False

    success, new_members = compute_membership_change(
        members,
        operation,
        target_member_id,
        project.get("owner_id")
    )
    if not success:
        _log_result(False, "membership_change_failed", executor_user_id, proj_id, operation, target_member_id)
        return False

    project["members"] = new_members
    _log_result(True, "ok", executor_user_id, proj_id, operation, target_member_id)
    return True
