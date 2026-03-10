from typing import Dict, Any

# Global dictionaries expected by log_project_membership_action.
# Populate these from elsewhere in your application.
USERS: Dict[int, Dict[str, Any]] = {}
PROJECTS: Dict[int, Dict[str, Any]] = {}


def log_project_membership_action(acting_user_id: int, prjt_id: int, member_action: str) -> bool:
    """
    Log a project membership action ('add' or 'remove') only if the acting user is the project owner.

    On success:
      - Prints a message stating the action was logged successfully and returns True.
    Otherwise:
      - Returns False without logging any action.
    """
    # Validate acting user exists
    user = USERS.get(acting_user_id)
    if not isinstance(user, dict):
        return False

    # Validate project exists
    project = PROJECTS.get(prjt_id)
    if not isinstance(project, dict):
        return False

    # Only the project owner can log membership actions
    if project.get("owner_id") != acting_user_id:
        return False

    # Validate action
    if member_action not in ("add", "remove"):
        return False

    print(f"[ACTION] Project {prjt_id}: member action '{member_action}' was logged successfully by owner {acting_user_id}")
    return True


def handle_project_membership(
    acting_user_id: int,
    prjt_id: int,
    member_action: str,
    target_member_id: int
) -> bool:
    """
    Enables project owners to add/remove members from their projects.
    Admins can manage membership for any project without restrictions.

    Returns True if the operation (add/remove) is successfully completed, False otherwise.
    """
    # Validate acting user exists
    user = USERS.get(acting_user_id)
    if not isinstance(user, dict):
        return False

    # Validate project exists
    project = PROJECTS.get(prjt_id)
    if not isinstance(project, dict):
        return False

    # Validate target member exists
    target_user = USERS.get(target_member_id)
    if not isinstance(target_user, dict):
        return False

    # Normalize and validate action
    action = member_action.lower() if isinstance(member_action, str) else None
    if action not in ("add", "remove"):
        return False

    # Access control: admin can manage any project, otherwise must be project owner
    is_admin = user.get("role") == "admin"
    is_owner = project.get("owner_id") == acting_user_id
    if not (is_admin or is_owner):
        return False

    # Validate members list
    members = project.get("members")
    if not isinstance(members, list):
        return False

    if action == "add":
        if target_member_id in members:
            return False
        members.append(target_member_id)
        return True

    # action == "remove"
    if target_member_id not in members:
        return False
    members.remove(target_member_id)
    return True
