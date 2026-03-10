from typing import Optional, Tuple, Dict, Any


def _get_data_stores() -> Optional[Tuple[Dict[int, dict], Dict[int, dict]]]:
    """Return (USERS, PROJECTS) if available, else None."""
    try:
        users = USERS  # type: ignore[name-defined]
        projects = PROJECTS  # type: ignore[name-defined]
        return users, projects
    except NameError:
        return None


def _normalize_action(member_action: str) -> Optional[str]:
    """Normalize and validate the action string."""
    if not isinstance(member_action, str):
        return None
    action = member_action.strip().lower()
    return action if action in ("add", "remove") else None


def _is_authorized(actor: Dict[str, Any], project: Dict[str, Any], acting_user_id: int) -> bool:
    """Check if actor can manage the given project."""
    return actor.get("role") == "admin" or acting_user_id == project.get("owner_id")


def _get_members(project: Dict[str, Any]) -> Optional[list]:
    """Fetch and validate the members list from a project."""
    members = project.get("members")
    return members if isinstance(members, list) else None


def _add_member(members: list, target_member_id: int, users: Dict[int, dict]) -> bool:
    """Add a member to the project if they exist and are not already a member."""
    if target_member_id not in users:
        return False
    if target_member_id in members:
        return True  # idempotent success
    members.append(target_member_id)
    return True


def _remove_member(project: Dict[str, Any], members: list, target_member_id: int) -> bool:
    """Remove a member from the project, preventing removal of the owner."""
    if target_member_id == project.get("owner_id"):
        return False
    if target_member_id not in members:
        return True  # idempotent success
    project["members"] = [uid for uid in members if uid != target_member_id]
    return True


def handle_project_membership(
    acting_user_id: int,
    prjt_id: int,
    member_action: str,
    target_member_id: int
) -> bool:
    """
    Manage project membership based on permissions.

    - Admins can add/remove members from any project.
    - Project owners can add/remove members from their own projects.
    - Returns True if the operation completes successfully (including no-op if state already satisfied), False otherwise.
    """
    stores = _get_data_stores()
    if stores is None:
        return False
    users, projects = stores

    project = projects.get(prjt_id)
    actor = users.get(acting_user_id)
    if project is None or actor is None:
        return False

    action = _normalize_action(member_action)
    if action is None:
        return False

    if not _is_authorized(actor, project, acting_user_id):
        return False

    members = _get_members(project)
    if members is None:
        return False

    if action == "add":
        return _add_member(members, target_member_id, users)
    else:
        return _remove_member(project, members, target_member_id)
