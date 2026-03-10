from typing import Dict, Any, List

# Note: The following globals are expected to be defined in the environment where this function runs:
# USERS: Dict[int, Dict[str, Any]]
# PROJECTS: Dict[int, Dict[str, Any]]

def handle_project_membership(
    active_user_id: int,
    proj_id: int,
    operation: str,
    target_member_id: int
) -> bool:
    """
    Allows project owners to add/remove members to their own projects, and admins to manage any project.

    Args:
        active_user_id: The ID of the user performing the action.
        proj_id: The ID of the project to modify.
        operation: Either 'add' or 'remove'.
        target_member_id: The user ID to add or remove from the project's members.

    Returns:
        True if the membership list was modified; otherwise False.
    """
    # Validate required globals
    try:
        users: Dict[int, Dict[str, Any]] = USERS  # type: ignore[name-defined]
        projects: Dict[int, Dict[str, Any]] = PROJECTS  # type: ignore[name-defined]
    except NameError:
        # Required globals not present
        return False

    # Validate existence of project and users
    active_user = users.get(active_user_id)
    if active_user is None:
        return False

    project = projects.get(proj_id)
    if project is None:
        return False

    # Ensure target user exists
    if target_member_id not in users:
        return False

    # Authorization: admin can manage any project; owner can manage own project
    is_admin = active_user.get("role") == "admin"
    is_owner = project.get("owner_id") == active_user_id
    if not (is_admin or is_owner):
        return False

    # Normalize members list
    members: List[int] = project.get("members") or []
    if not isinstance(members, list):
        # If somehow not a list, coerce to list to avoid runtime issues
        try:
            members = list(members)  # type: ignore[arg-type]
        except Exception:
            members = []
    project["members"] = members

    op = (operation or "").strip().lower()
    if op == "add":
        if target_member_id in members:
            return False  # no change
        members.append(target_member_id)
        return True
    elif op == "remove":
        if target_member_id not in members:
            return False  # no change
        members.remove(target_member_id)
        return True
    else:
        # Unsupported operation
        return False
