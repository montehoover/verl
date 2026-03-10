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
