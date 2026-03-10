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


def can_modify_project(executor_user_id: int, project: dict) -> bool:
    """
    Pure helper to determine if an executor can modify a given project.
    - Admins can modify any project.
    - Project owners can modify their own project.
    """
    executor = USERS.get(executor_user_id)
    if executor is None:
        return False
    is_admin = executor.get("role") == "admin"
    is_owner = project.get("owner_id") == executor_user_id
    return is_admin or is_owner


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

    if operation == "add":
        if target_member_id in new_members:
            return False, new_members
        new_members.append(target_member_id)
        return True, new_members

    if operation == "remove":
        if target_member_id not in new_members:
            return False, new_members
        new_members.remove(target_member_id)
        return True, new_members

    return False, new_members


def manage_team_access(executor_user_id: int, proj_id: int, operation: str, target_member_id: int) -> bool:
    """
    Manage project team access:
    - Admins can add/remove members on any project.
    - Project owners can add/remove members on their own projects only.

    Returns True if the requested operation successfully changed membership, otherwise False.
    """
    # Validate inputs
    if operation not in ("add", "remove"):
        return False

    # Check existence of executor, project, and target user
    executor = USERS.get(executor_user_id)
    if executor is None:
        return False

    project = PROJECTS.get(proj_id)
    if project is None:
        return False

    if target_member_id not in USERS:
        return False

    # Permission check via helper
    if not can_modify_project(executor_user_id, project):
        return False

    members = project.get("members")
    success, new_members = compute_membership_change(
        members,
        operation,
        target_member_id,
        project.get("owner_id")
    )
    if not success:
        return False

    # Apply the computed change
    project["members"] = new_members
    return True
