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

# Configure module-level logger for auditing membership operations
logger = logging.getLogger("project.membership")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


def can_manage_project(active_user_role: str, active_user_id: int, project_owner_id: int) -> bool:
    """
    Pure function: determines if a user can manage a project based on role or ownership.
    """
    if active_user_role == "admin":
        return True
    return project_owner_id == active_user_id


def apply_membership_operation(current_members, operation: str, target_member_id: int):
    """
    Pure function: returns (changed: bool, new_members: list[int]) without mutating current_members.
    Performs add/remove while ensuring idempotency.
    """
    op = operation.strip().lower() if isinstance(operation, str) else None
    new_members = list(current_members) if current_members is not None else []

    if op == "add":
        if target_member_id in new_members:
            return False, new_members
        new_members.append(target_member_id)
        return True, new_members

    if op == "remove":
        if target_member_id not in new_members:
            return False, new_members
        new_members.remove(target_member_id)
        return True, new_members

    # Invalid operation
    return False, new_members


def handle_project_membership(active_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Allows a project owner to add/remove members from their own projects.
    Admins can manage membership for any project.
    Returns True if a modification occurred, else False.
    """
    # Normalize operation for logging and validation
    op = operation.strip().lower() if isinstance(operation, str) else None

    # Log the attempt
    logger.info(
        "event=membership action=attempt user_id=%s project_id=%s operation=%s target_user_id=%s",
        active_user_id, proj_id, op, target_member_id
    )

    # Validate operation
    if op not in {"add", "remove"}:
        logger.warning(
            "event=membership action=denied reason=invalid_operation user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    # Validate existence of users and project
    active_user = USERS.get(active_user_id)
    if active_user is None:
        logger.warning(
            "event=membership action=denied reason=unknown_active_user user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    project = PROJECTS.get(proj_id)
    if project is None:
        logger.warning(
            "event=membership action=denied reason=unknown_project user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    # Authorization: admin or project owner
    if not can_manage_project(active_user.get("role"), active_user_id, project.get("owner_id")):
        logger.warning(
            "event=membership action=denied reason=insufficient_permissions user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    # Validate target member exists
    if target_member_id not in USERS:
        logger.warning(
            "event=membership action=denied reason=unknown_target_user user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    members = project.get("members")
    if not isinstance(members, list):
        logger.warning(
            "event=membership action=denied reason=invalid_members_structure user_id=%s project_id=%s operation=%s target_user_id=%s",
            active_user_id, proj_id, op, target_member_id
        )
        return False

    before_count = len(members)
    changed, new_members = apply_membership_operation(members, op, target_member_id)
    after_count = len(new_members)

    if not changed:
        reason = "already_member" if op == "add" else "not_a_member"
        logger.info(
            "event=membership action=noop reason=%s user_id=%s project_id=%s operation=%s target_user_id=%s members_before=%d members_after=%d",
            reason, active_user_id, proj_id, op, target_member_id, before_count, after_count
        )
        return False

    # Persist the new membership list
    project["members"] = new_members
    logger.info(
        "event=membership action=success user_id=%s project_id=%s operation=%s target_user_id=%s members_before=%d members_after=%d",
        active_user_id, proj_id, op, target_member_id, before_count, after_count
    )
    return True
