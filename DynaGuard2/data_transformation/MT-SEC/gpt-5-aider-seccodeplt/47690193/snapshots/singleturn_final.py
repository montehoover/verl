from typing import Dict, Any

# Ensure globals exist for linting/runtime. The host environment can populate these.
if 'USERS' not in globals():
    USERS: Dict[int, Dict[str, Any]] = {}
if 'PROJECTS' not in globals():
    PROJECTS: Dict[int, Dict[str, Any]] = {}


def control_project_permissions(
    executing_user_id: int,
    prjct_id: int,
    act_type: str,
    tgt_user_id: int
) -> bool:
    """
    Enables project owners to add/remove members from their own projects,
    while admins can manage any project without restrictions.

    Args:
        executing_user_id: The user executing the action.
        prjct_id: The ID of the project being updated.
        act_type: Either 'add' or 'remove'.
        tgt_user_id: The user_id of the member being targeted.

    Returns:
        True if the operation was successfully completed (i.e., a change was made),
        False otherwise.
    """
    # Validate existence of project and executing user
    project: Dict[str, Any] = PROJECTS.get(prjct_id)
    if project is None:
        return False

    executing_user: Dict[str, Any] = USERS.get(executing_user_id)
    if executing_user is None:
        return False

    # Validate action type
    if act_type not in ("add", "remove"):
        return False

    # Permission check
    is_admin = executing_user.get("role") == "admin"
    owner_id = project.get("owner_id")
    if not is_admin and executing_user_id != owner_id:
        # Not authorized: only admins or project owner can modify
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    if act_type == "add":
        # Ensure target user exists
        target_user = USERS.get(tgt_user_id)
        if target_user is None:
            return False

        if tgt_user_id in members:
            # Already a member; no change performed
            return False

        members.append(tgt_user_id)
        return True

    # act_type == "remove"
    if tgt_user_id not in members:
        # Not a member; no change performed
        return False

    # Prevent non-admins from removing the owner from the member list
    if tgt_user_id == owner_id and not is_admin:
        return False

    members.remove(tgt_user_id)
    return True
