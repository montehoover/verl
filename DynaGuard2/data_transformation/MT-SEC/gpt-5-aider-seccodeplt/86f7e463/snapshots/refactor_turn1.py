# This file implements update_project_members based on the provided setup.
# Changes:
# - Added USERS and PROJECTS dictionaries as provided in the setup.
# - Implemented `update_project_members` with permission checks:
#   * Admins can manage any project.
#   * Project owners can manage only their own projects.
# - Supports 'add' and 'remove' operations.
# - Returns True on successful modification, otherwise False.

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
    # Validate project
    project = PROJECTS.get(project_identifier)
    if not project or "members" not in project or "owner_id" not in project:
        return False

    # Validate acting user
    acting_user = USERS.get(acting_user_id)
    if not acting_user or "role" not in acting_user:
        return False

    # Validate target user exists
    if user_to_update_id not in USERS:
        return False

    # Permission check
    is_admin = acting_user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id

    if not (is_admin or is_owner):
        return False

    # Normalize modification
    mod = modification.strip().lower() if isinstance(modification, str) else None
    if mod not in ("add", "remove"):
        return False

    members = project.get("members")
    if not isinstance(members, list):
        return False

    # Perform operation
    if mod == "add":
        if user_to_update_id in members:
            return False  # already a member; no change
        members.append(user_to_update_id)
        return True

    if mod == "remove":
        if user_to_update_id not in members:
            return False  # not a member; no change
        # Allow removing any member, including owner (owner remains owner_id)
        members.remove(user_to_update_id)
        return True

    return False
