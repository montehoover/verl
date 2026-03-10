from typing import Any, Dict
from collections.abc import Mapping


def get_project_settings(project_key: str) -> Dict[str, Any]:
    """
    Retrieve the settings dictionary for the specified project.

    Assumes a global PROJECTS_DB is available in module globals().

    Behavior:
    - If PROJECTS_DB[project_key] is a dict and contains a 'settings' dict, return that.
    - Else if PROJECTS_DB[project_key] is a dict, return it (assumed to be the settings).
    - Else if the project object has a 'settings' attribute that is a dict, return it.
    - Raises KeyError if the project_key is not found.
    - Raises TypeError for invalid input or if settings are not in a dictionary form.

    :param project_key: The unique key (string) identifying the project.
    :return: The settings dictionary for the specified project.
    """
    if not isinstance(project_key, str):
        raise TypeError("project_key must be a string")

    db = globals().get("PROJECTS_DB")
    if db is None:
        raise NameError("PROJECTS_DB is not defined")
    if not isinstance(db, Mapping):
        raise TypeError("PROJECTS_DB must be a mapping/dictionary")

    try:
        project = db[project_key]
    except KeyError as exc:
        raise KeyError(f"Project with key '{project_key}' not found") from exc

    # Case 1: Conventional structure: {"settings": {...}}
    if isinstance(project, dict) and "settings" in project:
        settings = project["settings"]
        if isinstance(settings, dict):
            return settings
        raise TypeError("Project 'settings' must be a dictionary")

    # Case 2: The project entry itself is the settings dict
    if isinstance(project, dict):
        return project

    # Case 3: Object with a 'settings' attribute
    if hasattr(project, "settings"):
        settings = getattr(project, "settings")
        if isinstance(settings, dict):
            return settings
        raise TypeError("Project 'settings' attribute must be a dictionary")

    raise TypeError("Project settings not found or not in a dictionary form")


def check_user_role(member_id: str, project_key: str) -> bool:
    """
    Check whether the given member has a role that allows viewing or modifying settings.

    Returns True if:
    - The user is a member of the project; and
    - The user's role intersects with the set of roles allowed to access settings.

    The function attempts to handle a few common data layouts in PROJECTS_DB:
    - project["members"] can be:
        * dict: {member_id: role | {"role": "..."} | {"roles": [...]} | True}
        * list: [{"id": "...", "role": "..."}, {"member_id": "...", "roles": [...]}]
        * set: {"member_id1", "member_id2"}  (treated as role "member")
    - Alternative containers: project["team"], project["users"], project["memberships"]
    - Allowed roles can be specified in several places (first match is used):
        * project["permissions"]["settings_access_roles"]
        * project["settings"]["settings_access_roles"]
        * project["roles"]["settings"]
        * project["allowed_settings_roles"]
      If none found, a default allowlist is used: {"owner", "admin", "manager", "maintainer", "editor"}.

    :param member_id: The member's identifier (string).
    :param project_key: The project's key (string).
    :return: True if the member has sufficient role, otherwise False.
    """
    if not isinstance(member_id, str):
        raise TypeError("member_id must be a string")
    if not isinstance(project_key, str):
        raise TypeError("project_key must be a string")

    db = globals().get("PROJECTS_DB")
    if db is None or not isinstance(db, Mapping):
        # If the DB is missing or invalid, we cannot determine access.
        return False

    project = db.get(project_key)
    if project is None:
        return False

    def _as_role_set(value: Any) -> set[str]:
        roles: set[str] = set()
        if value is None:
            return roles
        if isinstance(value, str):
            roles.add(value.strip().lower())
            return roles
        if isinstance(value, (list, tuple, set)):
            for v in value:
                if isinstance(v, str):
                    roles.add(v.strip().lower())
        elif isinstance(value, dict):
            # Accept {role: True/False} style or {"role": "..."} or {"roles": [...]}
            if "role" in value and isinstance(value["role"], str):
                roles.add(value["role"].strip().lower())
            if "roles" in value and isinstance(value["roles"], (list, tuple, set)):
                for v in value["roles"]:
                    if isinstance(v, str):
                        roles.add(v.strip().lower())
            else:
                for k, v in value.items():
                    if isinstance(k, str) and bool(v):
                        roles.add(k.strip().lower())
        return roles

    def _extract_allowed_roles(obj: Any) -> set[str]:
        default_allowed = {"owner", "admin", "manager", "maintainer", "editor"}
        if isinstance(obj, dict):
            # Check common locations for settings access roles
            if isinstance(obj.get("permissions"), dict):
                roles = _as_role_set(obj["permissions"].get("settings_access_roles"))
                if roles:
                    return roles
            if isinstance(obj.get("settings"), dict):
                roles = _as_role_set(obj["settings"].get("settings_access_roles"))
                if roles:
                    return roles
            if isinstance(obj.get("roles"), dict):
                roles = _as_role_set(obj["roles"].get("settings"))
                if roles:
                    return roles
            roles = _as_role_set(obj.get("allowed_settings_roles"))
            if roles:
                return roles
        # Object-style attributes
        if hasattr(obj, "permissions"):
            roles = _as_role_set(getattr(obj, "permissions"))
            if roles:
                return roles
        if hasattr(obj, "settings"):
            roles = _as_role_set(getattr(obj, "settings"))
            if roles:
                return roles
        if hasattr(obj, "roles"):
            roles = _as_role_set(getattr(obj, "roles"))
            if roles:
                return roles
        return default_allowed

    def _extract_member_roles(obj: Any, uid: str) -> set[str]:
        containers = []
        # Dict style
        if isinstance(obj, dict):
            for key in ("members", "team", "users", "memberships"):
                if key in obj:
                    containers.append(obj[key])
        # Object style
        for key in ("members", "team", "users", "memberships"):
            if hasattr(obj, key):
                containers.append(getattr(obj, key))

        for container in containers:
            # Mapping: {member_id: role/roles/bool or nested dict}
            if isinstance(container, Mapping):
                if uid in container:
                    val = container[uid]
                    if isinstance(val, bool):
                        return {"member"} if val else set()
                    return _as_role_set(val) or {"member"}
            # List of member records
            if isinstance(container, list):
                for item in container:
                    if not isinstance(item, Mapping):
                        continue
                    if item.get("id") == uid or item.get("member_id") == uid or item.get("user_id") == uid:
                        # Prefer explicit role(s); otherwise treat as generic member
                        roles = _as_role_set(item.get("role")) | _as_role_set(item.get("roles")) | _as_role_set(item.get("permissions"))
                        return roles or {"member"}
            # Set of member IDs
            if isinstance(container, set):
                if uid in container:
                    return {"member"}
        return set()

    allowed_roles = _extract_allowed_roles(project)
    member_roles = _extract_member_roles(project, member_id)

    # Determine access by intersection of normalized role names
    return bool(allowed_roles & {r.lower() for r in member_roles})
