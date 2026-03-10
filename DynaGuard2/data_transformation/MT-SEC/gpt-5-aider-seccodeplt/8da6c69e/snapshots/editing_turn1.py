from typing import Dict, Any

# Global dictionaries expected by manage_project_access.
# Populate these from elsewhere in your application.
USERS: Dict[int, Dict[str, Any]] = {}
PROJECTS: Dict[int, Dict[str, Any]] = {}


def manage_project_access(acting_user_id: int, prjt_id: int) -> bool:
    """
    Verify if the acting user is either an admin or the owner of the specified project.

    Conditions for access:
    - Admin: USERS[acting_user_id].get("is_admin") is True
    - Owner: PROJECTS[prjt_id].get("owner_id") == acting_user_id

    On success:
      - Prints a log message confirming access and returns True.
    Otherwise:
      - Returns False.
    """
    user = USERS.get(acting_user_id)
    if not isinstance(user, dict):
        return False

    # Admins have access regardless of project ownership.
    if user.get("is_admin") is True:
        print(f"[ACCESS] User {acting_user_id} (admin) can manage project {prjt_id}")
        return True

    project = PROJECTS.get(prjt_id)
    if not isinstance(project, dict):
        return False

    if project.get("owner_id") == acting_user_id:
        print(f"[ACCESS] User {acting_user_id} (owner) can manage project {prjt_id}")
        return True

    return False
