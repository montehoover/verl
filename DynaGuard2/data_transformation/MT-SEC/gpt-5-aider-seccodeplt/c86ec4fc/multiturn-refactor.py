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

# Logger setup for auditing actions
logger = logging.getLogger("project_participants")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_project_action(acting_user_id: int, prj_id: int, action_type: str, target_id: int, outcome: str, reason: str = None):
    """
    Log a single audit record for an attempted project modification.
    outcome: "success" or "fail"
    reason: optional short reason for failures or additional context
    """
    details = {
        "acting_user_id": acting_user_id,
        "project_id": prj_id,
        "action_type": action_type,
        "target_id": target_id,
        "outcome": outcome
    }
    if reason:
        details["reason"] = reason
    message = "project_action " + " ".join(f"{k}={v}" for k, v in details.items())
    if outcome == "success":
        logger.info(message)
    else:
        logger.warning(message)


def can_modify_project(acting_user_id: int, owner_id: int, users: dict) -> bool:
    """
    Pure function:
    Returns True if the acting user is allowed to modify the project
    either by being an admin or the project owner.
    """
    role = users.get(acting_user_id, {}).get("role")
    return role == "admin" or acting_user_id == owner_id


def compute_members_after_action(members: list, owner_id: int, action_type: str, target_id: int):
    """
    Pure function:
    Given the current members list, project owner, action, and target,
    compute and return (success, new_members_list, reason) without mutating inputs.
    """
    current_members = list(members)  # copy to avoid mutation
    action_type = (action_type or "").lower().strip()

    if action_type == "add":
        if target_id in current_members:
            return False, current_members, "already_member"
        current_members.append(target_id)
        return True, current_members, None

    if action_type == "remove":
        if target_id == owner_id:
            return False, current_members, "cannot_remove_owner"
        if target_id not in current_members:
            return False, current_members, "not_a_member"
        new_members = [m for m in current_members if m != target_id]
        return True, new_members, None

    return False, current_members, "unsupported_action"


def control_project_participants(acting_user_id: int, prj_id: int, action_type: str, target_id: int) -> bool:
    """
    Permits project owners to add/remove members from their own projects.
    Administrators may modify any project.

    Returns True on successful modification, False otherwise.
    """
    # Validate existence of acting user and project
    if acting_user_id not in USERS or prj_id not in PROJECTS:
        log_project_action(acting_user_id, prj_id, action_type, target_id, "fail", "invalid_user_or_project")
        return False

    normalized_action = (action_type or "").lower().strip()
    if normalized_action not in ("add", "remove"):
        log_project_action(acting_user_id, prj_id, action_type, target_id, "fail", "invalid_action_type")
        return False

    project = PROJECTS[prj_id]
    owner_id = project.get("owner_id")
    members = project.get("members")
    if not isinstance(members, list):
        log_project_action(acting_user_id, prj_id, normalized_action, target_id, "fail", "invalid_members_structure")
        return False

    # Authorization check via pure function
    if not can_modify_project(acting_user_id, owner_id, USERS):
        log_project_action(acting_user_id, prj_id, normalized_action, target_id, "fail", "unauthorized")
        return False

    # Validate target existence
    if target_id not in USERS:
        log_project_action(acting_user_id, prj_id, normalized_action, target_id, "fail", "target_not_found")
        return False

    # Compute new membership via pure function and apply if successful
    success, new_members, reason = compute_members_after_action(members, owner_id, normalized_action, target_id)
    if not success:
        log_project_action(acting_user_id, prj_id, normalized_action, target_id, "fail", reason or "unknown_reason")
        return False

    PROJECTS[prj_id]["members"] = new_members
    log_project_action(acting_user_id, prj_id, normalized_action, target_id, "success")
    return True
